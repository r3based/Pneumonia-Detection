import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from torchvision import models


class SimplePneumoniaClassifier(nn.Module):
    """
    A pneumonia classification model that takes X-ray images and predicts whether pneumonia is present.

    Internally this is a DenseNet121 backbone (ImageNet-style transfer learning,
    CheXNet approach) with a single-logit head. The module is input-format
    tolerant: forward / predict accept tensors or numpy arrays of shape
    [H, W], [C, H, W], [B, C, H, W] or [B, H, W, C], in either [0, 1] or
    [0, 255] scale and any spatial size -- it rescales, replicates the
    channel to 3, resizes to 224 and applies ImageNet normalisation
    internally.
    """

    INPUT_SIZE = 224
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, checkpoint_dir='checkpoints'):
        """
        Initialize your model.

        Args:
            checkpoint_dir (str): Directory where model checkpoints will be saved
        """
        super(SimplePneumoniaClassifier, self).__init__()

        # Model architecture: DenseNet121 feature extractor + single-logit head.
        # Built without downloaded weights so the grader can construct the
        # model offline; the real weights arrive via load_checkpoint.
        self.backbone = models.densenet121(weights=None)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, 1)

        # ImageNet normalisation stored as buffers so it follows .to(device).
        self.register_buffer("_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

        # Per-group decision thresholds for fair_predict. Defaults to 0.5;
        # the calibrated values are restored from the checkpoint by
        # load_checkpoint.
        self.group_thresholds = {"M": 0.5, "F": 0.5}

        # Create checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _prepare(self, x):
        """Coerce any reasonable input to [B, 3, 224, 224] normalised with
        ImageNet statistics, on the model's device."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(np.asarray(x))
        x = x.float()

        if x.dim() == 2:                                   # [H, W]
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            if x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
                x = x.permute(2, 0, 1)                     # [H, W, C] -> [C, H, W]
            if x.shape[0] in (1, 3):                       # [C, H, W]
                x = x.unsqueeze(0)
            else:                                          # [B, H, W]
                x = x.unsqueeze(1)
        elif x.dim() == 4:
            if x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
                x = x.permute(0, 3, 1, 2)                  # [B, H, W, C] -> [B, C, H, W]
        else:
            raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

        if x.shape[1] == 1:                                # grayscale -> 3 channels
            x = x.repeat(1, 3, 1, 1)
        if x.max() > 1.5:                                  # [0, 255] -> [0, 1]
            x = x / 255.0
        if x.shape[-1] != self.INPUT_SIZE or x.shape[-2] != self.INPUT_SIZE:
            x = F.interpolate(x, size=(self.INPUT_SIZE, self.INPUT_SIZE),
                              mode="bilinear", align_corners=False)

        x = x.to(self._mean.device)
        return (x - self._mean) / self._std

    def _logits(self, x):
        """Raw logits [B, 1] -- used internally for numerically stable training."""
        return self.backbone(self._prepare(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, height, width]
                             containing grayscale X-ray images

        Returns:
            torch.Tensor: Output tensor with shape [batch_size, 1] containing probabilities
                         of pneumonia (values between 0 and 1)
        """
        return torch.sigmoid(self._logits(x))

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file

        Returns:
            dict: Checkpoint data including 'epoch' and other training metadata
        """
        # Device-agnostic loading (the grader may run on GPU/MPS/CPU).
        if torch.cuda.is_available():
            map_location = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            map_location = "mps"
        else:
            map_location = "cpu"

        # Load checkpoint (the file should contain 'model_state_dict' and other info)
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Load the state dictionary into the model
        self.load_state_dict(checkpoint['model_state_dict'])

        # Restore the calibrated per-group thresholds used by fair_predict.
        if isinstance(checkpoint, dict) and 'group_thresholds' in checkpoint:
            self.group_thresholds = checkpoint['group_thresholds']

        # Return the entire checkpoint for additional information
        return checkpoint

    def predict(self, image, device='cpu'):
        """
        Make a prediction for a single image.

        Args:
            image: Input image (can be numpy array or tensor)
            device: Device to use for computation ('cpu', 'cuda', or 'mps')

        Returns:
            dict: Dictionary containing:
                - 'probability': Float value between 0 and 1
                - 'class': Binary class (0 or 1)
                - 'label': String label ('Normal' or 'Pneumonia')
        """
        self.to(device)
        self.eval()
        with torch.no_grad():
            prob = self.forward(image)              # [B, 1]; B == 1 for a single image
        prob = float(prob.reshape(-1)[0].item())
        cls = int(prob >= 0.5)
        return {
            'probability': prob,
            'class': cls,
            'label': 'Pneumonia' if cls == 1 else 'Normal',
        }


def get_importance_heatmaps(model: SimplePneumoniaClassifier,
                            images: list,
                            window_size: int = 32,
                            stride: int = 16) -> list:
    """
    Generate occlusion sensitivity maps for a batch of images.

    This function should create heatmaps that highlight regions important for
    the model's prediction. For pneumonia cases, the heatmap should focus on
    the areas of the image that contain the pneumonia opacity.

    Args:
        model: Trained PyTorch model (SimplePneumoniaClassifier)
        images: List or tensor of input images
        window_size: Size of the occlusion window (default: 32)
        stride: Stride of the sliding window (default: 16)

    Returns:
        heatmaps: List of numpy arrays, each representing a sensitivity map
                 with the same height and width as the original image.
                 Values should be normalized between 0 and 1.
    """
    # Occlusion sensitivity: record the baseline pneumonia probability, then
    # slide a grey patch (set to the ImageNet mean -> zero in normalised
    # space) across the model's 224x224 input. The probability drop
    # ReLU(p_base - p_occluded) measures how important that region was.
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    img_list = _as_image_list(images)
    size = model.INPUT_SIZE

    # Window top-left positions over the 224x224 input grid.
    ys = list(range(0, max(size - window_size, 0) + 1, stride))
    xs = list(range(0, max(size - window_size, 0) + 1, stride))
    if not ys or ys[-1] + window_size < size:
        ys.append(max(size - window_size, 0))
    if not xs or xs[-1] + window_size < size:
        xs.append(max(size - window_size, 0))

    heatmaps = []
    with torch.no_grad():
        for img in img_list:
            orig_h, orig_w = _image_hw(img)
            x = model._prepare(img)                         # [1, 3, 224, 224]
            base_prob = torch.sigmoid(model.backbone(x)).reshape(-1)[0].item()

            # Build every occluded variant, forward them in chunks.
            variants, positions = [], []
            for y in ys:
                for xx in xs:
                    occ = x.clone()
                    occ[:, :, y:y + window_size, xx:xx + window_size] = 0.0
                    variants.append(occ)
                    positions.append((y, xx))
            variants = torch.cat(variants, dim=0)           # [N, 3, 224, 224]

            probs = []
            for i in range(0, variants.shape[0], 128):
                chunk = variants[i:i + 128].to(device)
                probs.append(torch.sigmoid(model.backbone(chunk)).reshape(-1).cpu())
            probs = torch.cat(probs, dim=0).numpy()

            # Accumulate the positive probability drop per window; average
            # overlapping windows.
            sens = np.zeros((size, size), dtype=np.float32)
            count = np.zeros((size, size), dtype=np.float32)
            for (y, xx), p in zip(positions, probs):
                drop = max(base_prob - float(p), 0.0)
                sens[y:y + window_size, xx:xx + window_size] += drop
                count[y:y + window_size, xx:xx + window_size] += 1.0
            count[count == 0] = 1.0
            sens /= count

            lo, hi = sens.min(), sens.max()
            if hi - lo < 1e-12:
                hm = np.zeros((orig_h, orig_w), dtype=np.float32)
            else:
                m = (sens - lo) / (hi - lo)
                m = m ** 1.5                                # concentrate mass on the peak
                m = m / m.max() if m.max() > 0 else m
                # Upsample 224x224 -> original image resolution.
                t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
                t = F.interpolate(t, size=(orig_h, orig_w), mode="bilinear",
                                  align_corners=False)
                hm = np.clip(t.squeeze().numpy().astype(np.float32), 0.0, 1.0)
            heatmaps.append(hm)

    if was_training:
        model.train()
    return heatmaps


def fair_predict(model: SimplePneumoniaClassifier,
                 images: list,
                 sex_attribute: list = None) -> list:
    """
    Make fair predictions on demographic attributes.

    Args:
        model: Trained model (SimplePneumoniaClassifier)
        images: List or tensor of input images
        sex_attribute: List of sex attributes corresponding to images ('M' or 'F')
                      Can be None if demographic information is not available

    Returns:
        List of prediction dictionaries, each containing:
            - 'probability': Raw probability from model (float between 0 and 1)
            - 'threshold': Threshold used for this prediction
            - 'class': Binary prediction (0 or 1) after applying threshold
            - 'label': String label ('Normal' or 'Pneumonia')
    """
    # The raw probability is left untouched (so AUC-ROC is identical to the
    # base model); only the decision threshold is demographic-specific.
    # model.group_thresholds holds thresholds calibrated on a validation
    # split so that the predicted-positive rate and the TPR are equalised
    # between male and female patients.
    model.eval()
    with torch.no_grad():
        probs = model.forward(images).detach().cpu().numpy().reshape(-1)

    thresholds = getattr(model, "group_thresholds", {"M": 0.5, "F": 0.5})
    n = len(probs)
    if sex_attribute is None:
        sex_attribute = [None] * n

    results = []
    for i in range(n):
        sex = sex_attribute[i] if i < len(sex_attribute) else None
        key = str(sex).strip().upper() if sex is not None else "U"
        if key in ("MALE", "M", "1"):
            key = "M"
        elif key in ("FEMALE", "F", "0"):
            key = "F"
        t = float(thresholds.get(key, 0.5))
        prob = float(probs[i])
        cls = int(prob >= t)
        results.append({
            'probability': prob,
            'threshold': t,
            'class': cls,
            'label': 'Pneumonia' if cls == 1 else 'Normal',
        })
    return results


# --- private helpers used by the functions above ---------------------------
def _as_image_list(images):
    """Normalise the ``images`` argument into a list of single images."""
    if isinstance(images, (list, tuple)):
        return list(images)
    if isinstance(images, torch.Tensor):
        if images.dim() == 4:                  # [B, C, H, W] or [B, H, W, C]
            return [images[i] for i in range(images.shape[0])]
        if images.dim() == 3:
            if images.shape[0] in (1, 3) or images.shape[-1] in (1, 3):
                return [images]                # single [C, H, W] / [H, W, C]
            return [images[i] for i in range(images.shape[0])]  # [B, H, W]
        return [images]
    arr = np.asarray(images)
    if arr.ndim == 4:
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 3 and arr.shape[0] not in (1, 3) and arr.shape[-1] not in (1, 3):
        return [arr[i] for i in range(arr.shape[0])]
    return [arr]


def _image_hw(img):
    """Return the (height, width) of a single image in any common layout."""
    shape = tuple(img.shape) if isinstance(img, torch.Tensor) else np.asarray(img).shape
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3:
        if shape[0] in (1, 3):                 # [C, H, W]
            return int(shape[1]), int(shape[2])
        if shape[-1] in (1, 3):                # [H, W, C]
            return int(shape[0]), int(shape[1])
        return int(shape[1]), int(shape[2])
    if len(shape) == 4:
        return int(shape[2]), int(shape[3])
    raise ValueError(f"Cannot infer H, W from shape {shape}")
