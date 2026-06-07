# src/evaluation/evaluate_generator_splits.py
"""
Per-generator evaluation of Dataset B test split.

For each of the 5 generators in Dataset B, builds a test set containing:
    - All real images from the Dataset B test split (label_b=0)
    - All AI images from that specific generator only

Evaluates the best checkpoint from a given experiment against each subset
and saves per-generator metrics to a combined CSV and heatmap.

Generator IDs (label_b):
    1 = SD 2.1
    2 = SDXL
    3 = SD 3
    4 = DALL-E 3
    5 = MidJourney v6

Usage (from project root):
    # Evaluate exp 13 (ResNet50) per generator
    python -m src.evaluation.evaluate_generator_splits \\
        --model resnet50 \\
        --experiment_name dataset_combined_v2_exp13 \\
        --save_metric val_f1

    # Evaluate exp 14 (ResNet18) per generator
    python -m src.evaluation.evaluate_generator_splits \\
        --model resnet18 \\
        --experiment_name dataset_combined_v2_exp14 \\
        --save_metric val_f1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from google.colab import drive

from sklearn.metrics import classification_report

from src.config import BATCH_SIZE, CHECKPOINTS_ROOT, DATASETS, LOGS_ROOT, NUM_CLASSES
from src.data.dataset import get_generator_test_loader
from src.evaluation._common import compute_metrics, print_metrics, run_inference
from src.training._common import find_best_checkpoint

GENERATOR_NAMES = {
    1: "SD 2.1",
    2: "SDXL",
    3: "SD 3",
    4: "DALL-E 3",
    5: "MidJourney v6",
}

MODEL_REGISTRY = {
    "resnet50": None,
    "resnet18": None,
}


def _register_models():
    from src.models.resnet50 import get_resnet50
    from src.models.resnet18 import get_resnet18
    MODEL_REGISTRY["resnet50"] = get_resnet50
    MODEL_REGISTRY["resnet18"] = get_resnet18


def _mount_drive() -> None:
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-generator evaluation of Dataset B")
    p.add_argument("--model", required=True, choices=["resnet50", "resnet18"],
                   help="Model architecture to evaluate")
    p.add_argument("--experiment_name", required=True,
                   help="Experiment name used during training (checkpoint path)")
    p.add_argument("--save_metric", default="val_f1", choices=["val_acc", "val_f1"],
                   help="Metric used to select best checkpoint (default: val_f1)")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--dog", action="store_true",
                   help="Apply DoG preprocessing. Must match training flag.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _mount_drive()
    _register_models()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt_dir  = os.path.join(CHECKPOINTS_ROOT, args.experiment_name, args.model)
    best_ckpt = find_best_checkpoint(ckpt_dir, args.model, metric=args.save_metric)

    model = MODEL_REGISTRY[args.model](num_classes=NUM_CLASSES).to(device)
    try:
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded: {Path(best_ckpt).name}  "
          f"val_acc={ckpt.get('val_acc', float('nan')):.4f}  "
          f"val_f1={ckpt.get('val_f1', float('nan')):.4f}")

    dataset_b_dir = DATASETS["dataset_b"]
    metadata_csv  = os.path.join(dataset_b_dir, "metadata.csv")

    # ── Evaluate per generator ─────────────────────────────────────────────────
    results = []

    for gen_id, gen_name in GENERATOR_NAMES.items():
        print(f"\n{'─'*60}")
        print(f"  Generator: {gen_name}  (label_b={gen_id})")
        print(f"{'─'*60}")

        loader, n_real, n_ai = get_generator_test_loader(
            dataset_b_dir  = dataset_b_dir,
            metadata_csv   = metadata_csv,
            generator_id   = gen_id,
            batch_size     = args.batch_size,
            already_resized= True,
            dog            = args.dog,
        )

        preds, labels = run_inference(model, loader, device)
        metrics = compute_metrics(preds, labels)
        print_metrics(metrics, args.model, args.experiment_name, f"dataset_b [{gen_name}]")

        print(f"\n  Per-class report:")
        print(classification_report(labels, preds, target_names=["ai", "real"], zero_division=0))

        results.append({
            "generator":  gen_name,
            "label_b":    gen_id,
            "n_real":     n_real,
            "n_ai":       n_ai,
            "accuracy":   metrics["accuracy"],
            "precision":  metrics["precision"],
            "recall":     metrics["recall"],
            "f1":         metrics["f1"],
        })

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # ── Save results ───────────────────────────────────────────────────────────
    output_dir = os.path.join(LOGS_ROOT, args.experiment_name, args.model, "generator_splits")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"{args.model}_dataset_b_generator_splits.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nMetrics saved → {csv_path}")
    print(df[["generator", "n_real", "n_ai", "accuracy", "f1"]].to_string(index=False))

    # ── Bar chart: F1 per generator ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#16A34A" if f >= 0.7 else "#DC2626" for f in df["f1"]]
    bars = ax.bar(df["generator"], df["f1"], color=colors, edgecolor="white", width=0.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro F1-score", fontsize=11)
    ax.set_xlabel("Generator", fontsize=11)
    ax.set_title(f"{args.model} — Dataset B per-generator F1  [{args.experiment_name}]",
                 fontsize=12, pad=10)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, val in zip(bars, df["f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()

    chart_path = os.path.join(output_dir, f"{args.model}_dataset_b_generator_f1.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved    → {chart_path}")

    print(f"\n{'═'*60}")
    print(f"Per-generator evaluation complete.")
    print(f"Outputs in: {output_dir}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()