# Pneumonia Detection — Model, Explainability & Fairness

Chest X-ray pneumonia assignment (RSNA Pneumonia Detection Challenge dataset).

## Files

| File | Role |
|------|------|
| [student_template.py](student_template.py) | **The submission.** Exactly the provided template, filled in — `SimplePneumoniaClassifier`, `get_importance_heatmaps`, `fair_predict`. Signatures unchanged. The grader only needs this file + `checkpoints/best_model.pt`. |
| [train.py](train.py) | Tooling only — **not submitted.** Loads the RSNA DICOMs, trains the model, calibrates fairness thresholds and writes `checkpoints/best_model.pt`. Also has an `evaluate` mode to self-check against the rubric. |
| `Dockerfile`, `docker-compose.yml`, `requirements.txt` | Run `train.py` in a container. |

`student_template.py` imports only `torch` / `torchvision` / `numpy`.
`train.py` imports it plus the training-only deps (`pandas`, `pydicom`,
`opencv`, `scikit-learn`).

## Approach

- **Task 1 — Classification.** `SimplePneumoniaClassifier` wraps a
  DenseNet121 backbone with a single-logit head (CheXNet-style transfer
  learning). `train.py` warm-starts the feature extractor with ImageNet
  weights — done in the trainer, not in `__init__`, so the submitted class
  keeps the exact template signature and stays offline-constructible for
  the grader. `forward` returns probabilities `[B, 1]`.
- **Task 2 — Heatmaps.** `get_importance_heatmaps` does occlusion
  sensitivity (as the template's `window_size` / `stride` args imply): a
  grey patch is slid over the 224×224 input, and the probability drop
  `ReLU(p_base − p_occluded)` scores each region. Overlapping windows are
  averaged, the map is mildly sharpened, upsampled to the original
  resolution and min-max normalised to `[0, 1]`.
- **Task 3 — Fairness.** `train.py` calibrates per-sex decision thresholds
  on a validation split to equalise predicted-positive rate and TPR, and
  bakes them into the checkpoint. `load_checkpoint` restores them;
  `fair_predict` leaves the raw probability untouched (so AUC-ROC is
  identical to the base model) and returns the group-specific `threshold`
  alongside each prediction.

## Dataset layout

Download the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
data and place it next to this repo so paths stay relative:

```
Pneumonia-Detection/
├── student_template.py
├── train.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── rsna-pneumonia-detection-challenge/
    ├── stage_2_train_labels.csv
    └── stage_2_train_images/
        └── <patientId>.dcm
```

## Run with Docker

Build:

```bash
docker build -t pneumonia-detection .
```

Train (writes `checkpoints/best_model.pt`):

```bash
docker run --rm --gpus all \
  -v "$(pwd)/rsna-pneumonia-detection-challenge:/workspace/rsna-pneumonia-detection-challenge" \
  -v "$(pwd)/checkpoints:/workspace/checkpoints" \
  pneumonia-detection \
  python train.py train
```

Self-evaluate against the grading rubric:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/rsna-pneumonia-detection-challenge:/workspace/rsna-pneumonia-detection-challenge" \
  -v "$(pwd)/checkpoints:/workspace/checkpoints" \
  pneumonia-detection \
  python train.py evaluate
```

Or via Compose (set `DATA_ROOT` if the dataset lives elsewhere):

```bash
docker compose run --rm train
docker compose run --rm evaluate
```

The image builds on `pytorch/pytorch:*-cuda12.1` — drop `--gpus all` to run
CPU-only (slower but functional); the code is device-agnostic
(CUDA → MPS → CPU).

### Smoke test (quick pipeline check)

```bash
python train.py train --max-train-batches 5 --epochs 1 --num-workers 0
```

## CLI reference (`train.py` only)

```
python train.py train     [--data-root DIR] [--epochs N] [--batch-size N]
                           [--lr LR] [--image-size N] [--no-pretrained]
                           [--max-train-batches N] [--num-workers N]
python train.py evaluate  [--data-root DIR] [--checkpoint PATH]
                           [--max-batches N] [--num-workers N]
```

## Checkpoint format

`checkpoints/best_model.pt` is saved exactly as the assignment requires:

```python
{
    "epoch": int,
    "model_state_dict": model.state_dict(),   # required key
    "optimizer_state_dict": optimizer.state_dict(),
    "auc": float,
    "group_thresholds": {"M": float, "F": float},  # consumed by fair_predict
}
```
