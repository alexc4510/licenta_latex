# src/config.py

import os

# ── Google Drive root ─────────────────────────────────────────────────────────

DRIVE_ROOT = "/content/drive/MyDrive/licenta"

# ── Datasets ──────────────────────────────────────────────────────────────────
# Keys are the canonical dataset identifiers used everywhere in the codebase.
# To add a new dataset: add one entry here and run the appropriate download/prepare
# scripts — everything else (training, evaluation, matrix) picks it up automatically.

DATASETS: dict[str, str] = {
    "ai_vs_human":         os.path.join(DRIVE_ROOT, "datasets", "ai_vs_human"),
    "dataset_b":           os.path.join(DRIVE_ROOT, "datasets", "dataset_b"),
    "dataset_c":           os.path.join(DRIVE_ROOT, "datasets", "dataset_c"),
    "dataset_b_balanced":  os.path.join(DRIVE_ROOT, "datasets", "dataset_b_balanced"),
    "dataset_combined":    os.path.join(DRIVE_ROOT, "datasets", "dataset_combined"),
    "dataset_combined_v2": os.path.join(DRIVE_ROOT, "datasets", "dataset_combined_v2"),
}

# ── Persistent output roots ───────────────────────────────────────────────────
# Structure:  <ROOT>/<dataset>/<model>/
# e.g.  checkpoints/dataset_b/resnet50/resnet50_epoch_03.pth

CHECKPOINTS_ROOT = os.path.join(DRIVE_ROOT, "checkpoints")
LOGS_ROOT        = os.path.join(DRIVE_ROOT, "logs")

# ── Training hyper-parameters ─────────────────────────────────────────────────

IMAGE_SIZE  = 224
BATCH_SIZE  = 128
EPOCHS      = 5
NUM_CLASSES = 2

# ImageNet normalisation — shared by all pretrained torchvision models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]