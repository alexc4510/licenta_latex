# src/training/train_resnet50.py
"""
Train ResNet-50 on a chosen dataset.

Checkpoints and logs are saved under:
    checkpoints/<experiment_name>/<model>/   resnet50_epoch_NN.pth
    logs/<experiment_name>/<model>/          training_metrics.csv  loss_curve.png  acc_curve.png

where <experiment_name> defaults to <dataset> if --experiment_name is not provided.

Backwards compatibility:
    All new flags have defaults that reproduce the exact behaviour of
    experiments 1-12. Running without any new flags is identical to before.

New flags for experiments 13/14:
    --augment              Extended data augmentation (color jitter, rotation, random erasing)
    --focal_loss           Use Focal Loss instead of weighted CrossEntropyLoss
    --focal_gamma          Focal loss gamma parameter (default: 2.0)
    --focal_alpha          Focal loss alpha parameter (default: None)
    --label_smoothing      Label smoothing factor (default: 0.0)
    --grad_clip            Gradient clipping max norm (default: None)
    --warmup_epochs        Linear LR warmup epochs (default: 0)
    --early_stopping       Early stopping patience in epochs (default: 0 = disabled)
    --save_metric          Metric for best checkpoint: val_acc or val_f1 (default: val_acc)
    --scheduler            LR scheduler: none or plateau (default: none)

Existing flags:
    --dog                  Difference of Gaussians preprocessing
    --experiment_name      Override checkpoint/log save path
    --resume               Resume from latest checkpoint

Usage (from project root in Colab):
    # Standard (experiments 1-12 behaviour)
    python -m src.training.train_resnet50 --dataset dataset_b

    # Full experiment 13 setup
    python -m src.training.train_resnet50 \\
        --dataset dataset_combined_v2 \\
        --epochs 20 \\
        --augment \\
        --focal_loss \\
        --label_smoothing 0.1 \\
        --grad_clip 1.0 \\
        --warmup_epochs 2 \\
        --early_stopping 5 \\
        --save_metric val_f1 \\
        --scheduler plateau \\
        --experiment_name dataset_combined_v2_exp13
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.optim as optim
from google.colab import drive

from src.config import BATCH_SIZE, CHECKPOINTS_ROOT, DATASETS, EPOCHS, LOGS_ROOT, NUM_CLASSES
from src.data.dataset import get_dataloaders
from src.models.resnet50 import get_resnet50
from src.training._common import (
    EarlyStopping, LinearWarmupScheduler,
    build_criterion, compute_class_weights,
    eval_epoch, load_latest_checkpoint,
    save_checkpoint, save_training_artifacts, train_epoch,
)

MODEL_NAME = "resnet50"


def _mount_drive() -> None:
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Train {MODEL_NAME}")

    # ── Existing flags (backwards compatible) ─────────────────────────────────
    p.add_argument("--dataset",          default="dataset_b", choices=list(DATASETS))
    p.add_argument("--epochs",           type=int, default=EPOCHS)
    p.add_argument("--batch_size",       type=int, default=BATCH_SIZE)
    p.add_argument("--not_resized",      action="store_true",
                   help="Images NOT pre-resized to 224 — apply Resize+CenterCrop at load time")
    p.add_argument("--resume",           action="store_true",
                   help="Resume from the latest checkpoint in the checkpoint directory")
    p.add_argument("--dog",              action="store_true",
                   help="Apply Difference of Gaussians preprocessing (sigma_weak=1.0, sigma_strong=2.0)")
    p.add_argument("--experiment_name",  default=None,
                   help="Override checkpoint/log save path to avoid overwriting previous results")

    # ── New flags (experiments 13/14) ─────────────────────────────────────────
    p.add_argument("--augment",          action="store_true",
                   help="Extended augmentation: ColorJitter + RandomRotation + RandomErasing")
    p.add_argument("--focal_loss",       action="store_true",
                   help="Use Focal Loss instead of weighted CrossEntropyLoss")
    p.add_argument("--focal_gamma",      type=float, default=2.0,
                   help="Focal loss gamma focusing parameter (default: 2.0)")
    p.add_argument("--focal_alpha",      type=float, default=None,
                   help="Focal loss alpha per-class weight (default: None)")
    p.add_argument("--label_smoothing",  type=float, default=0.0,
                   help="Label smoothing factor in [0, 1) (default: 0.0 = disabled)")
    p.add_argument("--grad_clip",        type=float, default=None,
                   help="Gradient clipping max norm (default: None = disabled)")
    p.add_argument("--warmup_epochs",    type=int, default=0,
                   help="Linear LR warmup epochs from 0 to base_lr (default: 0 = disabled)")
    p.add_argument("--early_stopping",   type=int, default=0,
                   help="Early stopping patience in epochs on val_f1 (default: 0 = disabled)")
    p.add_argument("--save_metric",      default="val_acc", choices=["val_acc", "val_f1"],
                   help="Metric used to select best checkpoint (default: val_acc)")
    p.add_argument("--scheduler",        default="none", choices=["none", "plateau"],
                   help="LR scheduler: none or plateau/ReduceLROnPlateau (default: none)")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _mount_drive()

    exp_name = args.experiment_name or args.dataset

    checkpoint_dir = os.path.join(CHECKPOINTS_ROOT, exp_name, MODEL_NAME)
    logs_dir       = os.path.join(LOGS_ROOT,        exp_name, MODEL_NAME)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir,       exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device={device}  |  model={MODEL_NAME}  |  dataset={args.dataset}")
    print(f"  experiment_name={exp_name}  |  dog={args.dog}  |  augment={args.augment}")
    print(f"  focal_loss={args.focal_loss}  |  label_smoothing={args.label_smoothing}")
    print(f"  grad_clip={args.grad_clip}  |  warmup_epochs={args.warmup_epochs}")
    print(f"  early_stopping={args.early_stopping}  |  save_metric={args.save_metric}")
    print(f"  scheduler={args.scheduler}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, _ = get_dataloaders(
        DATASETS[args.dataset],
        batch_size=args.batch_size,
        already_resized=not args.not_resized,
        dog=args.dog,
        augment=args.augment,
    )

    model     = get_resnet50(num_classes=NUM_CLASSES).to(device)
    weights   = compute_class_weights(train_loader, NUM_CLASSES, device)
    criterion = build_criterion(
        weights=weights,
        use_focal=args.focal_loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        label_smoothing=args.label_smoothing,
        num_classes=NUM_CLASSES,
    )
    # Initialize LR to 0 if warmup is enabled so the warmup scheduler
    # linearly increases it from 0 to 1e-4 over warmup_epochs.
    # Without warmup, start directly at 1e-4 as in experiments 1-12.
    base_lr = 1e-4
    init_lr = 0.0 if args.warmup_epochs > 0 else base_lr
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    # LR warmup
    warmup = LinearWarmupScheduler(optimizer, args.warmup_epochs, base_lr=base_lr) \
             if args.warmup_epochs > 0 else None

    # ReduceLROnPlateau scheduler
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    ) if args.scheduler == "plateau" else None

    # Early stopping
    early_stopper = EarlyStopping(patience=args.early_stopping) \
                    if args.early_stopping > 0 else None
    if early_stopper is not None and args.resume:
        early_stopper.counter    = es_state["counter"]
        early_stopper.best       = es_state["best"]
        print(f"  [EarlyStopping] Restored: counter={early_stopper.counter}  best={early_stopper.best:.4f}")

    start_epoch = 1
    es_state    = {"counter": 0, "best": -1.0}
    if args.resume:
        start_epoch, es_state = load_latest_checkpoint(checkpoint_dir, MODEL_NAME, model, optimizer, device)

    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
    }
    best_metric    = -1.0
    best_ckpt_path = ""

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'=' * 55}")
        print(f"  Epoch {epoch}/{args.epochs}  [{MODEL_NAME} | {exp_name}]")
        print(f"{'=' * 55}")
        t0 = time.time()

        # LR warmup step — only apply during warmup window
        # Guards against --resume starting at epoch > warmup_epochs
        # which would set LR above base_lr incorrectly
        if warmup is not None and epoch <= args.warmup_epochs:
            warmup.step(epoch)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=args.grad_clip,
        )
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)

        # ReduceLROnPlateau step (after warmup)
        if plateau is not None and (warmup is None or epoch > args.warmup_epochs):
            plateau.step(val_f1)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  [Scheduler] current lr={current_lr:.2e}")

        print(
            f"\n  Summary | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f} | "
            f"time={(time.time()-t0)/60:.1f}min"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        es_counter = early_stopper.counter if early_stopper is not None else 0
        es_best    = early_stopper.best    if early_stopper is not None else -1.0
        ckpt = save_checkpoint(
            model, optimizer, epoch, val_acc, checkpoint_dir, MODEL_NAME,
            val_f1=val_f1,
            early_stopping_counter=es_counter,
            early_stopping_best=es_best,
        )

        # Track best checkpoint based on chosen metric
        current_metric = val_f1 if args.save_metric == "val_f1" else val_acc
        if current_metric > best_metric:
            best_metric    = current_metric
            best_ckpt_path = ckpt
            print(f"  ★ New best  {args.save_metric}={best_metric:.4f}  →  {ckpt}")

        # Early stopping
        if early_stopper is not None:
            if early_stopper.step(val_f1):
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    print(f"\nTraining complete.  Best checkpoint: {best_ckpt_path}")
    save_training_artifacts(history, logs_dir, MODEL_NAME)


if __name__ == "__main__":
    main()