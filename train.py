"""Training / evaluation tooling for the pneumonia detector.

This file is NOT part of the submission -- the submission is the single
file ``student_template.py``. ``train.py`` only exists to produce the
``checkpoints/best_model.pt`` weights that the grader loads, and to let
you sanity-check the result against the grading rubric.

Usage
-----
    python train.py train     [--data-root ... --epochs ...]
    python train.py evaluate  [--data-root ... --checkpoint ...]

Dataset layout expected (relative paths only)::

    rsna-pneumonia-detection-challenge/
        stage_2_train_labels.csv
        stage_2_train_images/<patientId>.dcm

The checkpoint is written exactly in the format the assignment requires::

    {
        'epoch': int,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'auc': float,
        'group_thresholds': {'M': float, 'F': float},
    }
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from student_template import (SimplePneumoniaClassifier,
                              get_importance_heatmaps)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Data: RSNA DICOM loading
# ---------------------------------------------------------------------------
class PatientRecord:
    __slots__ = ("patient_id", "target", "sex", "boxes")

    def __init__(self, patient_id, target, sex, boxes):
        self.patient_id = patient_id
        self.target = target
        self.sex = sex
        self.boxes = boxes


def resize_image(img, size):
    """Resize a 2-D image to ``size x size`` (cv2 if available, else PIL)."""
    try:
        import cv2  # type: ignore
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def read_dicom(path):
    """Load a DICOM file -> (uint8 image, patient_sex)."""
    import pydicom
    ds = pydicom.dcmread(path)
    img = ds.pixel_array
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img -= img.min()
        m = img.max()
        if m > 0:
            img = img / m * 255.0
        img = img.astype(np.uint8)
    sex = getattr(ds, "PatientSex", "U") or "U"
    return img, sex


def load_records(labels_csv):
    """Collapse the labels CSV to one record per patient (keeping boxes)."""
    import pandas as pd
    df = pd.read_csv(labels_csv)
    records = []
    for pid, sub in df.groupby("patientId"):
        target = int(sub["Target"].iloc[0])
        boxes = []
        if target == 1:
            for _, r in sub.iterrows():
                if pd.notna(r["x"]):
                    boxes.append((float(r["x"]), float(r["y"]),
                                  float(r["width"]), float(r["height"])))
        records.append(PatientRecord(pid, target, "?", boxes))
    return records


def stratified_split(records, val_frac=0.1, seed=42):
    """Train/val split that preserves the positive ratio."""
    rng = np.random.default_rng(seed)
    pos = [r for r in records if r.target == 1]
    neg = [r for r in records if r.target == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos_val = int(len(pos) * val_frac)
    n_neg_val = int(len(neg) * val_frac)
    val = pos[:n_pos_val] + neg[:n_neg_val]
    train = pos[n_pos_val:] + neg[n_neg_val:]
    rng.shuffle(val)
    rng.shuffle(train)
    return train, val


def make_dataset_class():
    """Build the Dataset class (kept in a factory so torch.utils.data is the
    only thing imported eagerly here)."""
    from torch.utils.data import Dataset

    class RSNADataset(Dataset):
        def __init__(self, records, images_dir, image_size=224,
                     augment=False, return_boxes=False):
            self.records = records
            self.images_dir = images_dir
            self.image_size = image_size
            self.augment = augment
            self.return_boxes = return_boxes

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            rec = self.records[idx]
            path = os.path.join(self.images_dir, f"{rec.patient_id}.dcm")
            img, sex = read_dicom(path)
            rec.sex = sex
            orig_h, orig_w = img.shape[:2]
            img = resize_image(img, self.image_size)

            if self.augment:
                if np.random.rand() < 0.5:
                    img = np.ascontiguousarray(img[:, ::-1])
                if np.random.rand() < 0.5:
                    gamma = np.random.uniform(0.85, 1.15)
                    img = np.clip((img.astype(np.float32) / 255.0) ** gamma * 255.0,
                                  0, 255).astype(np.uint8)

            tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # [1,H,W]
            target = torch.tensor([float(rec.target)])                  # [1]

            if self.return_boxes:
                sx = self.image_size / orig_w
                sy = self.image_size / orig_h
                boxes = [(b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)
                         for b in rec.boxes]
                return tensor, target, sex, boxes
            return tensor, target, sex

    return RSNADataset


def make_balanced_sampler(records):
    """WeightedRandomSampler that draws roughly equal pos/neg."""
    targets = np.array([r.target for r in records])
    n_pos = max(int(targets.sum()), 1)
    n_neg = max(len(targets) - n_pos, 1)
    weights = np.where(targets == 1, 1.0 / n_pos, 1.0 / n_neg)
    return torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights), replacement=True)


def collate(batch):
    """Collate that keeps per-sample sex strings (and optional boxes) intact."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.stack([b[1] for b in batch], dim=0)
    sexes = [b[2] for b in batch]
    if len(batch[0]) == 4:
        boxes = [b[3] for b in batch]
        return imgs, targets, sexes, boxes
    return imgs, targets, sexes


# ---------------------------------------------------------------------------
# Fairness-threshold calibration
# ---------------------------------------------------------------------------
def calibrate_group_thresholds(scores, labels, sexes, default=0.5):
    """Search per-group thresholds minimising prediction-rate + TPR
    disparity, with a soft penalty against drifting far from 0.5.

    The disparity surface is piecewise-constant, so a grid over score
    mid-points plus a fine uniform grid finds a sharp optimum quickly.
    """
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    sexes = np.asarray(sexes)
    groups = [g for g in ("M", "F") if (sexes == g).sum() > 0]
    if len(groups) < 2:
        return {"M": default, "F": default}

    masks = {g: (sexes == g) for g in groups}
    base_grid = np.linspace(0.05, 0.95, 91)

    candidates = {}
    for g in groups:
        s = np.unique(scores[masks[g]])
        mids = (s[:-1] + s[1:]) / 2 if len(s) > 1 else s
        cand = np.unique(np.concatenate([mids, base_grid]))
        if len(cand) > 200:                    # keep the joint search small
            idx = np.linspace(0, len(cand) - 1, 200).astype(int)
            cand = cand[idx]
        candidates[g] = cand

    def pred_rate(g, t):
        return (scores[masks[g]] >= t).mean()

    def tpr(g, t):
        m = masks[g] & (labels == 1)
        return (scores[m] >= t).mean() if m.sum() else 0.0

    g1, g2 = groups[0], groups[1]
    best, best_score = (default, default), float("inf")
    for t1 in candidates[g1]:
        pr1, tpr1 = pred_rate(g1, t1), tpr(g1, t1)
        for t2 in candidates[g2]:
            pr_disp = abs(pr1 - pred_rate(g2, t2))
            tpr_disp = abs(tpr1 - tpr(g2, t2))
            shift_pen = (abs(0.5 - t1) + abs(0.5 - t2)) * 0.02
            score = pr_disp + 2.0 * tpr_disp + shift_pen
            if score < best_score:
                best_score, best = score, (t1, t2)

    out = {"M": default, "F": default}
    out[g1], out[g2] = float(best[0]), float(best[1])
    return out


# ---------------------------------------------------------------------------
# Train / evaluate loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optim, loss_fn, device, max_batches=None):
    model.train()
    total, total_loss = 0, 0.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        imgs, targets = batch[0].to(device), batch[1].to(device)
        optim.zero_grad(set_to_none=True)
        loss = loss_fn(model._logits(imgs), targets)       # BCEWithLogits on [B,1]
        loss.backward()
        optim.step()
        bs = targets.shape[0]
        total += bs
        total_loss += loss.item() * bs
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate_scores(model, loader, device, max_batches=None):
    """Collect probabilities / labels / sexes over a loader."""
    model.eval()
    scores, labels, sexes = [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        imgs = batch[0].to(device)
        probs = model.forward(imgs).cpu().numpy().reshape(-1)
        scores.extend(probs.tolist())
        labels.extend(batch[1].numpy().reshape(-1).astype(int).tolist())
        sexes.extend(list(batch[2]))
    return np.asarray(scores), np.asarray(labels), np.asarray(sexes)


def warm_start_backbone(model):
    """Warm-start the DenseNet feature extractor with ImageNet weights.

    Done here (not in ``SimplePneumoniaClassifier.__init__``) so the
    submitted model class keeps the exact template signature and stays
    offline-constructible for the grader.
    """
    try:
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
    except AttributeError:
        weights = None
    if weights is None:
        print("  (torchvision too old for ImageNet weights -- training from scratch)")
        return
    pre = models.densenet121(weights=weights)
    model.backbone.features.load_state_dict(pre.features.state_dict())
    del pre
    print("  warm-started DenseNet121 features with ImageNet weights")


def run_train(args):
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    print(f"Using device: {device}")

    labels_csv = os.path.join(args.data_root, "stage_2_train_labels.csv")
    images_dir = os.path.join(args.data_root, "stage_2_train_images")
    if not os.path.isfile(labels_csv):
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    print("Loading records...")
    records = load_records(labels_csv)
    train_rec, val_rec = stratified_split(records, args.val_frac, args.seed)
    print(f"Train: {len(train_rec)}  Val: {len(val_rec)}")

    RSNADataset = make_dataset_class()
    train_ds = RSNADataset(train_rec, images_dir, args.image_size, augment=True)
    val_ds = RSNADataset(val_rec, images_dir, args.image_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=make_balanced_sampler(train_rec),
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate)

    model = SimplePneumoniaClassifier(checkpoint_dir=args.checkpoint_dir)
    if not args.no_pretrained:
        warm_start_backbone(model)
    model = model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    best_auc = -1.0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optim, loss_fn,
                                     device, args.max_train_batches)
        scores, labels, sexes = evaluate_scores(model, val_loader, device)
        auc = roc_auc_score(labels, scores) if labels.sum() > 0 else 0.0
        thresholds = calibrate_group_thresholds(scores, labels, sexes)
        scheduler.step()

        print(f"[Epoch {epoch + 1}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  val_auc={auc:.4f}  "
              f"thresholds={thresholds}")

        if auc > best_auc:
            best_auc = auc
            model.group_thresholds = thresholds
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "auc": float(auc),
                "group_thresholds": thresholds,
            }, best_path)
            print(f"  -> new best checkpoint saved to {best_path}")

    print(f"\nBest validation AUC: {best_auc:.4f}\nBest checkpoint: {best_path}")


# ---------------------------------------------------------------------------
# Self-evaluation against the grading rubric
# ---------------------------------------------------------------------------
def bbox_coverage(heatmap, boxes, hm_h, hm_w):
    """Fraction of total heatmap mass that falls inside the bounding boxes."""
    if not boxes:
        return float("nan")
    mask = np.zeros((hm_h, hm_w), dtype=bool)
    for x, y, w, h in boxes:
        x0, y0 = max(int(x), 0), max(int(y), 0)
        x1, y1 = min(int(x + w), hm_w), min(int(y + h), hm_h)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    total = heatmap.sum()
    return float(heatmap[mask].sum() / total) if total > 0 else 0.0


def run_evaluate(args):
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    labels_csv = os.path.join(args.data_root, "stage_2_train_labels.csv")
    images_dir = os.path.join(args.data_root, "stage_2_train_images")
    records = load_records(labels_csv)
    _, val_rec = stratified_split(records, args.val_frac, args.seed)
    print(f"Val records: {len(val_rec)}")

    RSNADataset = make_dataset_class()
    ds = RSNADataset(val_rec, images_dir, args.image_size, return_boxes=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)

    model = SimplePneumoniaClassifier(checkpoint_dir=args.checkpoint_dir).to(device)
    model.load_checkpoint(args.checkpoint)
    print(f"Loaded {args.checkpoint}; group thresholds = {model.group_thresholds}")

    all_scores, all_labels, all_sexes, coverages = [], [], [], []
    for i, (imgs, targets, sexes, boxes) in enumerate(loader):
        if args.max_batches is not None and i >= args.max_batches:
            break
        imgs = imgs.to(device)
        probs = model.forward(imgs).detach().cpu().numpy().reshape(-1)
        all_scores.extend(probs.tolist())
        all_labels.extend(targets.numpy().reshape(-1).astype(int).tolist())
        all_sexes.extend(sexes)

        # Heatmap bbox-coverage for the positive cases in this batch.
        pos_idx = [j for j, t in enumerate(targets.numpy().reshape(-1)) if int(t) == 1]
        if pos_idx:
            hms = get_importance_heatmaps(model, imgs[pos_idx])
            for k, j in enumerate(pos_idx):
                coverages.append(bbox_coverage(hms[k], boxes[j],
                                               args.image_size, args.image_size))

    scores = np.asarray(all_scores)
    labels = np.asarray(all_labels)
    sexes = np.asarray(all_sexes)

    auc = roc_auc_score(labels, scores) if labels.sum() > 0 else float("nan")
    print(f"\nTask 1 - AUC-ROC: {auc:.4f}  (target >= 0.85)")

    if coverages:
        cov = float(np.nanmean(coverages))
        print(f"Task 2 - Avg heatmap bbox coverage: {cov * 100:.2f}%  (target >= 35%)")
    else:
        print("Task 2 - No positive cases evaluated.")

    thr = getattr(model, "group_thresholds", {"M": 0.5, "F": 0.5})
    preds = np.array([
        int(scores[i] >= thr.get(str(sexes[i]).upper()[:1], 0.5))
        for i in range(len(scores))
    ])
    print(f"\nTask 3 - AUC-ROC (unchanged by thresholds): {auc:.4f}  (target >= 0.8)")
    rates, tprs = {}, {}
    for g in ("M", "F"):
        m = sexes == g
        if m.sum() == 0:
            continue
        rates[g] = float(preds[m].mean())
        mp = m & (labels == 1)
        if mp.sum() > 0:
            tprs[g] = float(preds[mp].mean())
    print(f"  Prediction rates by group: {rates}")
    print(f"  TPRs by group:             {tprs}")
    if len(rates) == 2:
        print(f"  Prediction-rate disparity: {abs(rates['M'] - rates['F']):.4f}"
              f"  (target < 0.01)")
    if len(tprs) == 2:
        print(f"  TPR disparity:             {abs(tprs['M'] - tprs['F']):.4f}"
              f"  (target < 0.005)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Train / evaluate the pneumonia detector (tooling only -- "
                    "the submission is student_template.py).")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", default="rsna-pneumonia-detection-challenge",
                        help="Dir with stage_2_train_images/ and "
                             "stage_2_train_labels.csv.")
    common.add_argument("--batch-size", type=int, default=64)
    common.add_argument("--image-size", type=int, default=224)
    common.add_argument("--num-workers", type=int, default=4)
    common.add_argument("--val-frac", type=float, default=0.1)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--checkpoint-dir", default="checkpoints")

    pt = sub.add_parser("train", parents=[common], help="Train the model.")
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--weight-decay", type=float, default=1e-5)
    pt.add_argument("--no-pretrained", action="store_true",
                    help="Disable ImageNet warm-start (not recommended).")
    pt.add_argument("--max-train-batches", type=int, default=None,
                    help="Cap iters/epoch -- useful for smoke tests.")

    pe = sub.add_parser("evaluate", parents=[common],
                        help="Score a checkpoint against the rubric.")
    pe.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    pe.add_argument("--max-batches", type=int, default=None)

    return p


def main():
    args = build_parser().parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)


if __name__ == "__main__":
    main()
