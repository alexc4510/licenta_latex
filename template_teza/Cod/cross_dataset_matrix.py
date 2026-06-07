# src/evaluation/cross_dataset_matrix.py
"""
Cross-dataset generalisation matrix.

For a chosen model architecture, this script:
  1. Discovers every trained checkpoint under  checkpoints/<dataset>/<model>/
  2. For each checkpoint (= one "trained on" row), evaluates it against the
     test split of every available dataset (= one "tested on" column)
  3. Skips any (train, test) cell where either the checkpoint or the dataset
     directory does not yet exist — so the script remains valid when
     dataset_c is added later
  4. Outputs per metric (accuracy, f1, precision, recall):
       - A CSV table  (rows = trained_on, cols = tested_on)
       - A heatmap PNG

All outputs are saved to:
    logs/cross_dataset_matrix/<model_name>/

Usage (from project root):
    python -m src.evaluation.cross_dataset_matrix --model resnet50
    python -m src.evaluation.cross_dataset_matrix --model resnet18
    python -m src.evaluation.cross_dataset_matrix --model resnet50 --batch_size 64
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

from src.config import BATCH_SIZE, CHECKPOINTS_ROOT, DATASETS, LOGS_ROOT, NUM_CLASSES
from src.data.dataset import get_test_loader
from src.evaluation._common import compute_metrics, print_metrics, run_inference
from src.training._common import find_best_checkpoint

# Metrics to include in the output matrices
METRICS = ["accuracy", "f1", "precision", "recall"]

# Model name → constructor mapping
_MODEL_REGISTRY: dict[str, object] = {}

def _register_models() -> None:
    """Lazy import to avoid loading torch twice at module level."""
    from src.models.resnet50 import get_resnet50
    from src.models.resnet18 import get_resnet18
    _MODEL_REGISTRY["resnet50"] = get_resnet50
    _MODEL_REGISTRY["resnet18"] = get_resnet18
    try:
        from src.models.vit import get_vit
        _MODEL_REGISTRY["vit"] = get_vit
    except ImportError:
        pass  # vit.py not yet created — silently skip


def _mount_drive() -> None:
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-dataset evaluation matrix")
    p.add_argument(
        "--model", required=True, choices=["resnet50", "resnet18", "vit"],
        help="Model architecture to evaluate",
    )
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument(
        "--not_resized", action="store_true",
        help="Images are NOT pre-resized to 224 — apply Resize+CenterCrop at load time",
    )
    p.add_argument(
        "--save_metric", default="val_acc", choices=["val_acc", "val_f1"],
        help="Metric used to select best checkpoint per cell (default: val_acc). "
             "Use val_f1 when evaluating checkpoints from experiments 13/14.",
    )
    return p.parse_args()


# ── Cell runner ───────────────────────────────────────────────────────────────

def _evaluate_cell(
    train_dataset: str,
    test_dataset: str,
    model_name: str,
    batch_size: int,
    already_resized: bool,
    device: str,
    save_metric: str = "val_acc",
) -> dict[str, float] | None:
    """
    Load the best checkpoint for (train_dataset, model_name), run it against
    the test split of test_dataset, return the metrics dict.

    Returns None and prints a warning if either:
      - the checkpoint directory / files don't exist
      - the test dataset directory doesn't exist
    """
    checkpoint_dir = os.path.join(CHECKPOINTS_ROOT, train_dataset, model_name)
    test_data_dir  = DATASETS.get(test_dataset, "")

    # ── Guard: checkpoint must exist ─────────────────────────────────────────
    if not os.path.isdir(checkpoint_dir):
        print(f"  [skip] No checkpoint dir for trained={train_dataset}: {checkpoint_dir}")
        return None

    try:
        best_ckpt = find_best_checkpoint(checkpoint_dir, model_name, metric=save_metric)
    except FileNotFoundError as exc:
        print(f"  [skip] {exc}")
        return None

    # ── Guard: test dataset must exist ────────────────────────────────────────
    test_split_dir = os.path.join(test_data_dir, "test")
    if not os.path.isdir(test_split_dir):
        print(f"  [skip] Test split not found for {test_dataset}: {test_split_dir}")
        return None

    # ── Load model ────────────────────────────────────────────────────────────
    constructor = _MODEL_REGISTRY[model_name]
    model = constructor(num_classes=NUM_CLASSES).to(device)
    ckpt  = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"\n  ┌ trained={train_dataset}  →  tested={test_dataset}\n"
        f"  │ checkpoint: {Path(best_ckpt).name}"
        f"  (val_acc={ckpt.get('val_acc', float('nan')):.4f})"
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    test_loader = get_test_loader(
        test_data_dir,
        batch_size=batch_size,
        already_resized=already_resized,
    )
    preds, labels = run_inference(model, test_loader, device)
    metrics       = compute_metrics(preds, labels)
    print_metrics(metrics, model_name, train_dataset, test_dataset)

    # Free GPU memory before the next cell
    del model
    torch.cuda.empty_cache()

    return metrics


# ── Matrix builder ────────────────────────────────────────────────────────────

def build_matrix(
    model_name: str,
    batch_size: int,
    already_resized: bool,
    device: str,
    save_metric: str = "val_acc",
) -> dict[str, pd.DataFrame]:
    """
    Iterate over all (train_dataset × test_dataset) combinations.
    Return a dict mapping metric_name → DataFrame of shape
    (n_train_datasets × n_test_datasets).
    """
    dataset_keys = list(DATASETS.keys())

    # results[train_ds][test_ds] = {accuracy: ..., f1: ..., ...} | None
    results: dict[str, dict[str, dict[str, float] | None]] = {}

    for train_ds in dataset_keys:
        results[train_ds] = {}
        for test_ds in dataset_keys:
            print(f"\n{'─'*60}")
            print(f"  Cell: trained_on={train_ds}  |  tested_on={test_ds}")
            print(f"{'─'*60}")
            results[train_ds][test_ds] = _evaluate_cell(
                train_ds, test_ds, model_name,
                batch_size, already_resized, device,
                save_metric=save_metric,
            )

    # Build one DataFrame per metric
    matrices: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        data = {}
        for train_ds in dataset_keys:
            row = {}
            for test_ds in dataset_keys:
                cell = results[train_ds][test_ds]
                row[test_ds] = cell[metric] if cell is not None else np.nan
            data[train_ds] = row
        # DataFrame: index = train_ds (row), columns = test_ds (col)
        matrices[metric] = pd.DataFrame(data).T

    return matrices


# ── Output: CSV + heatmap ─────────────────────────────────────────────────────

def save_matrix_outputs(
    matrices: dict[str, pd.DataFrame],
    output_dir: str,
    model_name: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for metric, df in matrices.items():
        # CSV
        csv_path = os.path.join(output_dir, f"{model_name}_{metric}_matrix.csv")
        df.to_csv(csv_path, float_format="%.4f")
        print(f"\n  {metric} matrix saved → {csv_path}")
        print(df.to_string(float_format=lambda x: f"{x:.4f}"))

        # Heatmap PNG
        # Replace NaN with -1 for visualisation and mark them distinctly
        display_df = df.copy()
        mask       = display_df.isna()

        fig, ax = plt.subplots(figsize=(max(4, len(df.columns) * 1.8),
                                        max(3, len(df) * 1.5)))

        sns.heatmap(
            display_df.fillna(-1),
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            vmin=0.0, vmax=1.0,
            mask=mask,          # NaN cells rendered as grey below
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            annot_kws={"size": 10},
        )

        # Grey out missing cells
        if mask.any().any():
            sns.heatmap(
                display_df.fillna(-1),
                annot=pd.DataFrame(
                    [["N/A" if mask.iloc[r, c] else ""
                      for c in range(mask.shape[1])]
                     for r in range(mask.shape[0])],
                    index=df.index, columns=df.columns,
                ),
                fmt="",
                cmap=matplotlib.colors.ListedColormap(["#cccccc"]),
                mask=~mask,
                cbar=False,
                linewidths=0.5,
                linecolor="white",
                ax=ax,
                annot_kws={"size": 10, "color": "black"},
            )

        ax.set_xlabel("Tested on", fontsize=11)
        ax.set_ylabel("Trained on", fontsize=11)
        ax.set_title(
            f"{model_name} — {metric.capitalize()} generalisation matrix",
            fontsize=13, pad=12,
        )
        # Rotate axis labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        png_path = os.path.join(output_dir, f"{model_name}_{metric}_matrix.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Heatmap saved → {png_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    _mount_drive()
    _register_models()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice={device}  |  model={args.model}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Datasets in scope: {list(DATASETS.keys())}")
    print(f"Checkpoint root:   {CHECKPOINTS_ROOT}")

    matrices   = build_matrix(
        model_name=args.model,
        batch_size=args.batch_size,
        already_resized=not args.not_resized,
        device=device,
        save_metric=args.save_metric,
    )
    n          = len(DATASETS)
    output_dir = os.path.join(LOGS_ROOT, "cross_dataset_matrix", args.model, f"{n}x{n}")
    save_matrix_outputs(matrices, output_dir, args.model)

    print(f"\n{'═'*60}")
    print(f"Matrix complete.  Outputs in: {output_dir}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()