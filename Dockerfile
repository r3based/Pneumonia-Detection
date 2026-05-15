# syntax=docker/dockerfile:1.4
#
# Pneumonia-Detection training image.
#
# Builds on the official PyTorch CUDA runtime so training picks up a GPU
# automatically when ``--gpus all`` is passed to ``docker run``.  On
# CPU-only hosts the same image still works (it just runs slower).
#
# Example
# -------
# Build:
#     docker build -t pneumonia-detection .
#
# Train (with GPU and the RSNA dataset mounted into the container):
#     docker run --rm --gpus all \
#         -v $(pwd)/rsna-pneumonia-detection-challenge:/workspace/rsna-pneumonia-detection-challenge \
#         -v $(pwd)/checkpoints:/workspace/checkpoints \
#         pneumonia-detection \
#         python train.py train
#
# Evaluate:
#     docker run --rm --gpus all \
#         -v $(pwd)/rsna-pneumonia-detection-challenge:/workspace/rsna-pneumonia-detection-challenge \
#         -v $(pwd)/checkpoints:/workspace/checkpoints \
#         pneumonia-detection \
#         python train.py evaluate
#
# The submission itself is the single file student_template.py (the three
# template deliverables, nothing else).  train.py is tooling only -- it
# produces checkpoints/best_model.pt and is not submitted.

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV needs libgl1; pydicom is pure-python but benefits from gdcm for
# compressed transfer-syntaxes that occasionally appear in DICOM files.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy and install dependencies first to keep the layer cache warm.
COPY requirements.txt /workspace/requirements.txt
# Skip torch/torchvision in requirements.txt because the base image already
# ships compatible CUDA-enabled wheels; reinstalling would replace them
# with CPU versions.
RUN pip install --no-cache-dir \
        $(grep -Ev '^(torch|torchvision)' requirements.txt)

# Now copy the source.  Place this after pip install so editing Python
# code doesn't bust the dependency layer.
COPY . /workspace

# Ensure the checkpoints directory exists even if no volume is mounted.
RUN mkdir -p /workspace/checkpoints

# Default command — print the help so a bare ``docker run`` is informative.
CMD ["python", "train.py", "--help"]
