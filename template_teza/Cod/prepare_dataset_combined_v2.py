# scripts/prepare_dataset_combined_v2.py
"""
Creates dataset_combined_v2 by merging subsets of Dataset A, Dataset B and
Dataset C into a single balanced training and validation set.

Key design decisions:
    - Dataset B images are semantically paired: each real image is matched with
      its AI counterpart from the same caption, per generator. This maximises
      the difficulty of content-based shortcuts for the model.
    - MidJourney v6 is excluded from Dataset B to reduce overlap with Dataset A
      (both use diffusion-based generation; MidJourney is the most distinct
      generator in Dataset B and its exclusion keeps the paradigm separation cleaner).
    - Dataset C is included at a smaller proportion since GAN detection is
      already solved by previous experiments and does not need more data.
    - All splits are perfectly balanced at 1:1 real/AI.

Construction logic:
    Train (80,000 images):
        Dataset A:  23,000 real  +  23,000 AI   (random sample from 31,980)
        Dataset B:   7,000 real  +   7,000 AI   (semantically paired, 1,750 per generator x 4)
        Dataset C:  10,000 real  +  10,000 AI   (random sample from 50,000)
        Total:      40,000 real  +  40,000 AI   = 80,000

    Val (13,000 images):
        Dataset A:   3,000 real  +   3,000 AI   (random sample from 3,997)
        Dataset B:   1,500 real  +   1,500 AI   (semantically paired, 375 per generator x 4)
        Dataset C:   2,000 real  +   2,000 AI   (random sample from 10,000)
        Total:       6,500 real  +   6,500 AI   = 13,000

    Test:
        No test split. Dummy test/ai/ and test/real/ folders are created with
        one placeholder image each to prevent get_dataloaders() from crashing
        during training (it always loads all three splits). Evaluation is
        performed directly against the original test splits of Dataset A, B
        and C using the --checkpoint_dataset flag.

Prerequisites:
    - datasets/ai_vs_human/        (run download_dataset_a.py + prepare_dataset_a.py)
    - datasets/dataset_b/          (run download_dataset_b.py)
    - datasets/dataset_b/metadata.csv  (run build_dataset_b_metadata.py)
    - datasets/dataset_c/          (run download_dataset_c.py)

Run once. Re-running is safe — already-copied images are skipped.

Usage:
    python scripts/prepare_dataset_combined_v2.py
"""

from __future__ import annotations

import csv
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from google.colab import drive
from PIL import Image
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────

DRIVE_ROOT   = "/content/drive/MyDrive/licenta"
SRC_A        = os.path.join(DRIVE_ROOT, "datasets", "ai_vs_human")
SRC_B        = os.path.join(DRIVE_ROOT, "datasets", "dataset_b")
SRC_C        = os.path.join(DRIVE_ROOT, "datasets", "dataset_c")
DST_DIR      = os.path.join(DRIVE_ROOT, "datasets", "dataset_combined_v2")
METADATA_CSV = os.path.join(SRC_B, "metadata.csv")

RANDOM_SEED  = 42

# MidJourney v6 (label_b=5) is excluded — see module docstring for rationale
GENERATORS = [1, 2, 3, 4]   # SD2.1, SDXL, SD3, DALL-E3

LABEL_B_MAP = {
    0: "real",
    1: "SD2.1",
    2: "SDXL",
    3: "SD3",
    4: "DALL-E3",
    5: "MidJourney_v6",  # excluded
}

TARGETS = {
    "train": {
        "a_real": 23000, "a_ai": 23000,
        "b_real": 7000,  "b_ai_per_generator": 1750,   # 1750 x 4 = 7000
        "c_real": 10000, "c_ai": 10000,
    },
    "val": {
        "a_real": 3000,  "a_ai": 3000,
        "b_real": 1500,  "b_ai_per_generator": 375,    # 375 x 4 = 1500
        "c_real": 2000,  "c_ai": 2000,
    },
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _mount_drive() -> None:
    if not os.path.isdir("/content/drive/MyDrive"):
        print("Mounting Google Drive ...")
        drive.mount("/content/drive")
    else:
        print("Google Drive already mounted.")


def _make_dirs() -> None:
    for split in ["train", "val"]:
        for label in ["ai", "real"]:
            os.makedirs(os.path.join(DST_DIR, split, label), exist_ok=True)
    # Dummy test split to prevent get_dataloaders crash during training
    for label in ["ai", "real"]:
        folder = os.path.join(DST_DIR, "test", label)
        os.makedirs(folder, exist_ok=True)
        dummy_path = os.path.join(folder, "dummy.png")
        if not os.path.exists(dummy_path):
            Image.new("RGB", (224, 224), color=0).save(dummy_path)


def _load_metadata() -> list[dict]:
    records = []
    with open(METADATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label_a"] = int(row["label_a"])
            row["label_b"] = int(row["label_b"])
            records.append(row)
    return records


def _list_pngs(folder: str) -> list[Path]:
    """List PNG files using os.listdir to avoid Drive glob indexing issues."""
    p = Path(folder)
    if not p.is_dir():
        return []
    return sorted([p / f for f in os.listdir(folder) if f.lower().endswith(".png")])


def _count(folder: str) -> int:
    return len(_list_pngs(folder))


def _copy_file(src: str, dst: str) -> bool:
    """Copy src to dst. Returns True if copied, False if already existed."""
    if os.path.exists(dst):
        return False
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as exc:
        print(f"\n  [!] Failed to copy {src}: {exc}")
        return False


def _copy_batch(
    src_paths: list[str],
    dst_folder: str,
    desc: str,
    prefix: str = "",
) -> tuple[int, int]:
    """
    Copy a list of source files to dst_folder.
    prefix is prepended to the filename to avoid cross-dataset name collisions.
    Returns (copied, skipped).
    """
    copied = skipped = 0
    for src in tqdm(src_paths, desc=desc, unit="img"):
        dst = os.path.join(dst_folder, prefix + Path(src).name)
        if _copy_file(src, dst):
            copied += 1
        else:
            skipped += 1
    return copied, skipped


# ── DATASET B SEMANTIC PAIRING ────────────────────────────────────────────────

def _build_b_paired(
    metadata: list[dict],
    b_split: str,
    real_target: int,
    ai_per_generator: int,
) -> tuple[list[str], list[str]]:
    """
    Build semantically paired real and AI image lists from Dataset B.

    Pairing logic:
        - Real images are sorted by filename (sequential index = MS-COCO order)
        - For each generator, AI images are sorted by filename
        - Real images [i*n : (i+1)*n] are paired with AI images [0:n] from
          generator i, where n = ai_per_generator
        - This ensures each real image appears alongside its caption-matched
          AI counterpart from one generator

    Returns (real_paths, ai_paths) — both lists have the same length.
    """
    split_records = [r for r in metadata if r["split"] == b_split]

    # Real records sorted by filename index
    real_records = sorted(
        [r for r in split_records if r["label_a"] == 0],
        key=lambda r: r["filename"]
    )

    # AI records per generator, sorted by filename index
    ai_by_gen: dict[int, list[dict]] = defaultdict(list)
    for r in split_records:
        if r["label_a"] == 1 and r["label_b"] in GENERATORS:
            ai_by_gen[r["label_b"]].append(r)
    for gen in GENERATORS:
        ai_by_gen[gen].sort(key=lambda r: r["filename"])

    real_paths = []
    ai_paths   = []

    for i, gen in enumerate(GENERATORS):
        # Take the i-th block of real images
        start = i * ai_per_generator
        end   = start + ai_per_generator
        real_block = real_records[start:end]
        ai_block   = ai_by_gen[gen][:ai_per_generator]

        if len(real_block) < ai_per_generator:
            print(f"  [warn] B real block {i}: only {len(real_block)} available, needed {ai_per_generator}")
        if len(ai_block) < ai_per_generator:
            print(f"  [warn] B gen {gen} ({LABEL_B_MAP[gen]}): only {len(ai_block)} available, needed {ai_per_generator}")

        for r in real_block:
            real_paths.append(os.path.join(SRC_B, r["split"], "real", r["filename"]))
        for r in ai_block:
            ai_paths.append(os.path.join(SRC_B, r["split"], "ai", r["filename"]))

    # Verify total matches target
    total_b_real = len(GENERATORS) * ai_per_generator
    assert len(real_paths) == total_b_real, f"Expected {total_b_real} real, got {len(real_paths)}"
    assert len(ai_paths)   == total_b_real, f"Expected {total_b_real} AI, got {len(ai_paths)}"

    return real_paths, ai_paths


# ── SPLIT BUILDER ─────────────────────────────────────────────────────────────

def _build_split(
    split_name: str,
    metadata: list[dict],
    a_split: str,
    b_split: str,
    c_split: str,
    targets: dict,
) -> None:
    print(f"\n{'─'*60}")
    print(f"  Building: {split_name}")
    print(f"{'─'*60}")

    dst_real = os.path.join(DST_DIR, split_name, "real")
    dst_ai   = os.path.join(DST_DIR, split_name, "ai")

    rng = random.Random(RANDOM_SEED)
    total_copied = total_skipped = 0

    # ── Dataset A — real ──────────────────────────────────────────────────────
    a_real_pool = _list_pngs(os.path.join(SRC_A, a_split, "real"))
    a_real_sel  = rng.sample(a_real_pool, min(targets["a_real"], len(a_real_pool)))
    print(f"\n  [A/real]  pool={len(a_real_pool):,}  selected={len(a_real_sel):,}")
    c, s = _copy_batch([str(p) for p in a_real_sel], dst_real,
                       desc=f"  {split_name}/real [A]", prefix="a_")
    total_copied += c; total_skipped += s

    # ── Dataset A — AI ────────────────────────────────────────────────────────
    a_ai_pool = _list_pngs(os.path.join(SRC_A, a_split, "ai"))
    a_ai_sel  = rng.sample(a_ai_pool, min(targets["a_ai"], len(a_ai_pool)))
    print(f"\n  [A/ai]    pool={len(a_ai_pool):,}  selected={len(a_ai_sel):,}")
    c, s = _copy_batch([str(p) for p in a_ai_sel], dst_ai,
                       desc=f"  {split_name}/ai   [A]", prefix="a_")
    total_copied += c; total_skipped += s

    # ── Dataset B — semantically paired ───────────────────────────────────────
    print(f"\n  [B]  Building semantically paired real+AI (4 generators, no MidJourney):")
    print(f"       {targets['b_ai_per_generator']} real + {targets['b_ai_per_generator']} AI per generator")
    b_real_paths, b_ai_paths = _build_b_paired(
        metadata, b_split,
        real_target=targets["b_real"],
        ai_per_generator=targets["b_ai_per_generator"],
    )
    for i, gen in enumerate(GENERATORS):
        start = i * targets["b_ai_per_generator"]
        end   = start + targets["b_ai_per_generator"]
        print(f"       gen {gen} ({LABEL_B_MAP[gen]:<10}): "
              f"real[{start}:{end}] paired with AI[0:{targets['b_ai_per_generator']}]")

    print(f"\n  [B/real]  selected={len(b_real_paths):,}")
    c, s = _copy_batch(b_real_paths, dst_real,
                       desc=f"  {split_name}/real [B]", prefix="b_")
    total_copied += c; total_skipped += s

    print(f"\n  [B/ai]    selected={len(b_ai_paths):,}")
    c, s = _copy_batch(b_ai_paths, dst_ai,
                       desc=f"  {split_name}/ai   [B]", prefix="b_")
    total_copied += c; total_skipped += s

    # ── Dataset C — real ──────────────────────────────────────────────────────
    c_real_pool = _list_pngs(os.path.join(SRC_C, c_split, "real"))
    c_real_sel  = rng.sample(c_real_pool, min(targets["c_real"], len(c_real_pool)))
    print(f"\n  [C/real]  pool={len(c_real_pool):,}  selected={len(c_real_sel):,}")
    c, s = _copy_batch([str(p) for p in c_real_sel], dst_real,
                       desc=f"  {split_name}/real [C]", prefix="c_")
    total_copied += c; total_skipped += s

    # ── Dataset C — AI ────────────────────────────────────────────────────────
    c_ai_pool = _list_pngs(os.path.join(SRC_C, c_split, "ai"))
    c_ai_sel  = rng.sample(c_ai_pool, min(targets["c_ai"], len(c_ai_pool)))
    print(f"\n  [C/ai]    pool={len(c_ai_pool):,}  selected={len(c_ai_sel):,}")
    c, s = _copy_batch([str(p) for p in c_ai_sel], dst_ai,
                       desc=f"  {split_name}/ai   [C]", prefix="c_")
    total_copied += c; total_skipped += s

    # ── Split summary ─────────────────────────────────────────────────────────
    final_real = _count(dst_real)
    final_ai   = _count(dst_ai)
    print(f"\n  {split_name} complete:")
    print(f"    real:  {final_real:,}  (A:{len(a_real_sel):,} + B:{len(b_real_paths):,} + C:{len(c_real_sel):,})")
    print(f"    AI:    {final_ai:,}  (A:{len(a_ai_sel):,} + B:{len(b_ai_paths):,} + C:{len(c_ai_sel):,})")
    print(f"    total: {final_real + final_ai:,}  |  copied={total_copied:,}  skipped={total_skipped:,}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _mount_drive()

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    missing = []
    for path, label in [
        (SRC_A,        "datasets/ai_vs_human"),
        (SRC_B,        "datasets/dataset_b"),
        (SRC_C,        "datasets/dataset_c"),
        (METADATA_CSV, "datasets/dataset_b/metadata.csv"),
    ]:
        if not os.path.exists(path):
            missing.append(f"  x  {label}  ->  {path}")
    if missing:
        raise FileNotFoundError(
            "The following required paths are missing:\n" + "\n".join(missing) + "\n"
            "Run the appropriate download/prepare scripts first."
        )

    print(f"\n{'='*60}")
    print(f"  dataset_combined_v2 construction")
    print(f"{'='*60}")
    print(f"  Source A:    {SRC_A}")
    print(f"  Source B:    {SRC_B}  (generators: {[LABEL_B_MAP[g] for g in GENERATORS]})")
    print(f"  Source C:    {SRC_C}")
    print(f"  Destination: {DST_DIR}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"\n  Target sizes:")
    t = TARGETS["train"]
    print(f"    train  real: {t['a_real'] + t['b_real'] + t['c_real']:,}"
          f"  (A:{t['a_real']:,} + B:{t['b_real']:,} + C:{t['c_real']:,})")
    print(f"    train  AI:   {t['a_ai'] + t['b_ai_per_generator']*len(GENERATORS) + t['c_ai']:,}"
          f"  (A:{t['a_ai']:,} + B:{t['b_ai_per_generator']*len(GENERATORS):,} + C:{t['c_ai']:,})")
    v = TARGETS["val"]
    print(f"    val    real: {v['a_real'] + v['b_real'] + v['c_real']:,}"
          f"  (A:{v['a_real']:,} + B:{v['b_real']:,} + C:{v['c_real']:,})")
    print(f"    val    AI:   {v['a_ai'] + v['b_ai_per_generator']*len(GENERATORS) + v['c_ai']:,}"
          f"  (A:{v['a_ai']:,} + B:{v['b_ai_per_generator']*len(GENERATORS):,} + C:{v['c_ai']:,})")

    _make_dirs()

    print(f"\n  Loading dataset_b metadata ...")
    metadata = _load_metadata()
    print(f"  Loaded {len(metadata):,} records.")

    # ── Build splits ──────────────────────────────────────────────────────────
    _build_split(
        split_name="train",
        metadata=metadata,
        a_split="train",
        b_split="train",
        c_split="train",
        targets=TARGETS["train"],
    )

    _build_split(
        split_name="val",
        metadata=metadata,
        a_split="val",
        b_split="val",
        c_split="val",
        targets=TARGETS["val"],
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE. Final dataset_combined_v2 counts:")
    grand_total = 0
    for split in ["train", "val", "test"]:
        for label in ["ai", "real"]:
            folder = os.path.join(DST_DIR, split, label)
            n = _count(folder)
            grand_total += n
            print(f"    {split}/{label}: {n:,}")
    print(f"    grand total (incl. dummy test): {grand_total:,}")
    print(f"\n  No real test split was created.")
    print(f"  Evaluate using --checkpoint_dataset dataset_combined_v2 against:")
    print(f"    --dataset ai_vs_human    (~8,000 images, diffusion SD v1.5)")
    print(f"    --dataset dataset_b      (45,000 images, diffusion 4 models, unbalanced)")
    print(f"    --dataset dataset_c      (20,000 images, GAN StyleGAN)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()