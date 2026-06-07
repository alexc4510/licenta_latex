# src/training/_common.py
"""
Shared training/evaluation loop logic.
Imported by train_resnet18.py and train_resnet50.py — never run directly.

New in experiments 13/14:
    - Focal loss with optional label smoothing
    - Gradient clipping
    - Linear LR warmup + ReduceLROnPlateau scheduler
    - Early stopping on val F1
    - Checkpoint saving on val F1 instead of val acc

All new features are opt-in via arguments passed from the training scripts.
Default values reproduce the exact behaviour of experiments 1-12.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for binary and multi-class classification.

    Down-weights easy examples (high confidence correct predictions) and
    focuses training on hard examples (low confidence or misclassified).
    Particularly effective when the model over-predicts the majority class.

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher values down-weight easy examples more
        strongly. gamma=0 reduces to standard cross-entropy. Default: 2.0
        (standard value from Lin et al. 2017, RetinaNet paper).
    alpha : float
        Optional per-class weighting scalar in [0, 1]. When provided,
        the positive class (AI) is weighted by alpha and the negative class
        (real) by 1-alpha. Set to None to disable. Default: None.
    label_smoothing : float
        Smoothing factor applied to hard labels before computing focal loss.
        Prevents overconfidence during long training runs. 0.0 = no smoothing
        (standard cross-entropy behaviour). Default: 0.0.
    reduction : str
        'mean' or 'sum'. Default: 'mean'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        label_smoothing: float = 0.0,
        num_classes: int = 2,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.label_smoothing = label_smoothing
        self.num_classes     = num_classes
        self.reduction       = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(inputs)
                smooth_targets.fill_(self.label_smoothing / (self.num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            log_prob = F.log_softmax(inputs, dim=1)
            ce_loss  = -(smooth_targets * log_prob).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Focal weighting
        prob = torch.exp(-ce_loss)
        focal_loss = (1 - prob) ** self.gamma * ce_loss

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(
    train_loader: DataLoader,
    num_classes: int,
    device: str,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training set.

    Formula:  weight[i] = total_samples / (num_classes * count[i])

    For a balanced dataset all weights come out to 1.0 — no effect on loss.
    For an imbalanced dataset minority classes get higher weights, penalising
    the model more for misclassifying them.

    Works with any ImageFolder-based DataLoader via .dataset.targets.
    Prints a per-class breakdown so the values are always visible in the log.
    """
    targets      = torch.tensor(train_loader.dataset.targets)
    total        = len(targets)
    weights      = torch.zeros(num_classes)
    idx_to_class = {v: k for k, v in train_loader.dataset.class_to_idx.items()}

    for c in range(num_classes):
        count      = (targets == c).sum().item()
        weights[c] = total / (num_classes * count) if count > 0 else 1.0
        print(f"  class '{idx_to_class[c]}' (idx={c}): count={count}  weight={weights[c]:.4f}")

    return weights.to(device)


def build_criterion(
    weights: torch.Tensor,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    label_smoothing: float = 0.0,
    num_classes: int = 2,
) -> nn.Module:
    """
    Build the loss function based on the provided arguments.

    use_focal=False, label_smoothing=0.0 → standard CrossEntropyLoss with
    inverse-frequency weights (identical to experiments 1-12).

    use_focal=True → FocalLoss with optional label smoothing. Class weights
    are ignored when focal loss is used since focal loss handles imbalance
    dynamically through the gamma focusing parameter.

    Parameters
    ----------
    weights         : inverse-frequency class weights (used only if not focal)
    use_focal       : if True, use FocalLoss instead of CrossEntropyLoss
    focal_gamma     : focusing parameter for FocalLoss (default: 2.0)
    focal_alpha     : per-class weight for FocalLoss (default: None)
    label_smoothing : smoothing factor (used in FocalLoss if use_focal=True,
                      or in CrossEntropyLoss if use_focal=False and > 0)
    num_classes     : number of output classes
    """
    if use_focal:
        print(f"  Loss: FocalLoss(gamma={focal_gamma}, alpha={focal_alpha}, "
              f"label_smoothing={label_smoothing})")
        return FocalLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            label_smoothing=label_smoothing,
            num_classes=num_classes,
        )
    else:
        print(f"  Loss: CrossEntropyLoss(label_smoothing={label_smoothing})")
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


# ── LR Warmup ─────────────────────────────────────────────────────────────────

class LinearWarmupScheduler:
    """
    Linear learning rate warmup over a fixed number of epochs.

    Linearly increases LR from 0 to base_lr over warmup_epochs.
    After warmup, hands control to ReduceLROnPlateau.

    Parameters
    ----------
    optimizer     : the optimizer whose LR is being warmed up
    warmup_epochs : number of epochs to warm up over
    base_lr       : target LR at the end of warmup
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        base_lr: float,
    ):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr       = base_lr

    def step(self, epoch: int) -> None:
        """Call at the start of each epoch (1-indexed)."""
        if epoch <= self.warmup_epochs:
            lr = self.base_lr * epoch / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
            print(f"  [Warmup] Epoch {epoch}/{self.warmup_epochs}  lr={lr:.2e}")


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    log_every: int = 50,
    grad_clip: float | None = None,
) -> tuple[float, float]:
    """
    One full training pass. Returns (avg_loss, accuracy).

    grad_clip — if provided, clips gradient norm to this value before
                each optimizer step. Prevents exploding gradients during
                longer training runs. Default: None (no clipping).
    """
    model.train()
    total_loss = 0.0
    correct = total = 0
    start = time.time()

    for i, (images, labels) in enumerate(loader):
        t_batch = time.time()
        images  = images.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        if i % log_every == 0:
            done = i + 1
            eta  = (len(loader) - done) * (time.time() - start) / done
            print(
                f"  Batch {done:>4}/{len(loader)} | "
                f"loss={loss.item():.4f} | "
                f"batch={time.time()-t_batch:.2f}s | "
                f"ETA={eta/60:.1f}min"
            )

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, float]:
    """
    One full evaluation pass. Returns (avg_loss, accuracy, macro_f1).

    macro_f1 is computed over all batches and used for early stopping
    and checkpoint saving when --save_metric f1 is passed.
    """
    model.eval()
    total_loss = 0.0
    correct = total = 0
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images  = images.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item()
        preds = outputs.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), correct / total, macro_f1


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Early stopping based on a monitored metric (higher is better).

    Stops training if the metric does not improve for `patience` consecutive
    epochs. Used with val F1 as the monitored metric.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after last improvement before stopping.
        Default: 5. Set to 0 or None to disable early stopping.
    """

    def __init__(self, patience: int = 5):
        self.patience   = patience
        self.best       = -1.0
        self.counter    = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """
        Call after each epoch with the monitored metric value.
        Returns True if training should stop.
        """
        if self.patience is None or self.patience <= 0:
            return False

        if metric > self.best:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs "
                  f"(best={self.best:.4f})")
            if self.counter >= self.patience:
                print(f"  [EarlyStopping] Stopping training.")
                self.should_stop = True

        return self.should_stop


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    checkpoint_dir: str,
    model_name: str,
    val_f1: float = -1.0,
    early_stopping_counter: int = 0,
    early_stopping_best: float = -1.0,
) -> str:
    """
    Save a full checkpoint that includes val_acc, val_f1 and epoch metadata.
    Returns the saved path.

    File format:  <model_name>_epoch_<NN>.pth
    Stored dict:
        epoch                int
        val_acc              float
        val_f1               float  (new — used for F1-based checkpointing)
        model_state_dict     OrderedDict
        optimizer_state_dict OrderedDict
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch:02d}.pth")
    torch.save(
        {
            "epoch":                    epoch,
            "val_acc":                  val_acc,
            "val_f1":                   val_f1,
            "early_stopping_counter":   early_stopping_counter,
            "early_stopping_best":      early_stopping_best,
            "model_state_dict":         model.state_dict(),
            "optimizer_state_dict":     optimizer.state_dict(),
        },
        path,
    )
    return path


def find_best_checkpoint(
    checkpoint_dir: str,
    model_name: str,
    metric: str = "val_acc",
) -> str:
    """
    Scan *checkpoint_dir* for files matching <model_name>_epoch_*.pth,
    read the stored metric inside each, and return the path of the
    checkpoint with the highest value.

    metric — 'val_acc' (default, experiments 1-12) or 'val_f1'
             (experiments 13-14). Falls back to val_acc if val_f1
             is not found in older checkpoints.

    Raises FileNotFoundError if no valid checkpoints are found.
    """
    candidates = sorted(Path(checkpoint_dir).glob(f"{model_name}_epoch_*.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints matching '{model_name}_epoch_*.pth' in:\n  {checkpoint_dir}"
        )

    best_path: Path | None = None
    best_val = -1.0

    for p in candidates:
        try:
            try:
                ckpt = torch.load(p, map_location="cpu", weights_only=True)
            except Exception:
                ckpt = torch.load(p, map_location="cpu", weights_only=False)
            val  = float(ckpt.get(metric, ckpt.get("val_acc", -1.0)))
            if val > best_val:
                best_val  = val
                best_path = p
        except Exception as exc:
            print(f"  [warn] Skipping unreadable checkpoint {p.name}: {exc}")

    if best_path is None:
        raise FileNotFoundError(
            f"Found checkpoint files but none could be read in:\n  {checkpoint_dir}"
        )

    print(f"  Best checkpoint: {best_path.name}  ({metric}={best_val:.4f})")
    return str(best_path)


def load_latest_checkpoint(
    checkpoint_dir: str,
    model_name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> int:
    """
    Load the most recently saved checkpoint (highest epoch number) from
    *checkpoint_dir* into *model* and *optimizer*.
    Returns the epoch number to resume from (last completed epoch + 1).
    Returns 1 if no checkpoint is found (fresh start).
    """
    candidates = sorted(Path(checkpoint_dir).glob(f"{model_name}_epoch_*.pth"))
    if not candidates:
        print("  No checkpoint found — starting from scratch.")
        return 1, {"counter": 0, "best": -1.0}

    latest = candidates[-1]
    try:
        try:
            ckpt = torch.load(latest, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        epoch = int(ckpt["epoch"])
        print(f"  Resumed from: {latest.name}  "
              f"(epoch={epoch}  val_acc={ckpt.get('val_acc', float('nan')):.4f}  "
              f"val_f1={ckpt.get('val_f1', float('nan')):.4f})")
        es_state = {
            "counter": ckpt.get("early_stopping_counter", 0),
            "best":    ckpt.get("early_stopping_best", -1.0),
        }
        return epoch + 1, es_state
    except Exception as exc:
        print(f"  [warn] Could not load checkpoint {latest.name}: {exc} — starting from scratch.")
        return 1, {"counter": 0, "best": -1.0}


# ── Training artifacts ────────────────────────────────────────────────────────

def save_training_artifacts(
    history: dict[str, list[Any]],
    logs_dir: str,
    model_name: str,
) -> None:
    """Save training history CSV and loss/accuracy/f1 curve PNGs to *logs_dir*."""
    os.makedirs(logs_dir, exist_ok=True)
    df = pd.DataFrame(history)

    csv_path = os.path.join(logs_dir, f"{model_name}_training_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved:  {csv_path}")

    plots = [("loss", "Loss"), ("acc", "Accuracy")]
    if "val_f1" in df.columns:
        plots.append(("f1", "F1 Score"))

    for metric, title in plots:
        fig, ax = plt.subplots()
        if f"train_{metric}" in df.columns:
            ax.plot(df["epoch"], df[f"train_{metric}"], label="Train")
        ax.plot(df["epoch"], df[f"val_{metric}"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{model_name} — {title}")
        ax.legend()
        path = os.path.join(logs_dir, f"{model_name}_{metric}_curve.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved:     {path}")