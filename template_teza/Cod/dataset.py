# src/data/dataset.py

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def _safe_num_workers() -> int:
    """
    Cap at 2 workers for Colab. Drive I/O + more workers = frequent
    DataLoader crashes. Falls back to 0 if cpu_count is unavailable.
    """
    return min(2, os.cpu_count() or 1)


class DifferenceOfGaussians:
    """
    Difference of Gaussians (DoG) transform.

    Applies two Gaussian blurs with different standard deviations to the
    input image and returns their difference. The result highlights edges
    and high-frequency artefacts while suppressing low-frequency content.

    Applied AFTER ToTensor() so the input is a float tensor in [0, 1].
    The output is normalized back to [0, 1] before ImageNet normalization.

    Parameters
    ----------
    sigma_weak   : float
        Standard deviation of the weak (less blurry) Gaussian.
        Default: 1.0  →  kernel 7x7  (covers 3σ in each direction)
    sigma_strong : float
        Standard deviation of the strong (more blurry) Gaussian.
        Default: 2.0  →  kernel 13x13

    Kernel size formula (standard in literature):
        kernel_size = 2 * ceil(3 * sigma) + 1
    """

    def __init__(self, sigma_weak: float = 1.0, sigma_strong: float = 2.0):
        import math
        self.sigma_weak   = sigma_weak
        self.sigma_strong = sigma_strong

        def _kernel(sigma: float) -> int:
            return 2 * math.ceil(3 * sigma) + 1

        self.blur_weak   = transforms.GaussianBlur(kernel_size=_kernel(sigma_weak),   sigma=sigma_weak)
        self.blur_strong = transforms.GaussianBlur(kernel_size=_kernel(sigma_strong), sigma=sigma_strong)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        weak   = self.blur_weak(tensor)
        strong = self.blur_strong(tensor)
        dog    = weak - strong
        dog    = dog - dog.min()
        max_val = dog.max()
        if max_val > 0:
            dog = dog / max_val
        return dog

    def __repr__(self) -> str:
        return (f"DifferenceOfGaussians("
                f"sigma_weak={self.sigma_weak}, sigma_strong={self.sigma_strong})")


def _build_transforms(
    already_resized: bool = True,
    dog: bool = False,
    augment: bool = False,
):
    """
    Return (train_transform, eval_transform).

    Parameters
    ----------
    already_resized : bool
        True  → images are exactly IMAGE_SIZExIMAGE_SIZE on disk.
                 Skip Resize/CenterCrop — saves meaningful I/O time.
        False → apply the standard Resize(256) → CenterCrop(224).

    dog : bool
        True  → apply Difference of Gaussians after ToTensor and before
                 normalization. Extracts edge/contour information.
                 Must be used consistently in both training and evaluation.
        False → standard pipeline, no DoG preprocessing.

    augment : bool
        True  → apply extended augmentation during training:
                 random horizontal flip (always applied regardless of this flag)
                 + ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
                 + RandomRotation(degrees=10)
                 + RandomErasing(p=0.2, scale=(0.02, 0.1))
                 Used for experiments 13/14 to prevent overfitting and
                 discourage dataset-specific shortcut learning.
        False → only random horizontal flip (behaviour of experiments 1-12).

    Note: RandomErasing is applied after ToTensor since it operates on tensors.
    All spatial augmentations are applied before ToTensor.
    """
    spatial = [] if already_resized else [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
    ]

    spatial_aug = [transforms.RandomHorizontalFlip()]
    if augment:
        spatial_aug += [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(degrees=10),
        ]

    to_tensor = [transforms.ToTensor()]

    dog_transform = [DifferenceOfGaussians(sigma_weak=1.0, sigma_strong=2.0)] if dog else []

    tensor_aug = [transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))] if augment else []

    normalise = [transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]

    train_tf = transforms.Compose(
        spatial + spatial_aug + to_tensor + dog_transform + tensor_aug + normalise
    )
    eval_tf = transforms.Compose(
        spatial + to_tensor + dog_transform + normalise
    )
    return train_tf, eval_tf


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    already_resized: bool = True,
    num_workers: int | None = None,
    dog: bool = False,
    augment: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from an ImageFolder layout:

        data_dir/
            train/  ai/  real/
            val/    ai/  real/
            test/   ai/  real/

    ImageFolder assigns labels alphabetically → ai=0, real=1.
    This is consistent across all datasets so no remapping is needed.

    Parameters
    ----------
    dog     : apply Difference of Gaussians preprocessing to all splits.
    augment : apply extended augmentation to the train split only.
              Eval and test splits are never augmented beyond DoG.
    """
    train_tf, eval_tf = _build_transforms(already_resized, dog=dog, augment=augment)
    nw = num_workers if num_workers is not None else _safe_num_workers()

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=eval_tf)

    assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx, (
        f"class_to_idx mismatch between splits in {data_dir}.\n"
        f"  train={train_ds.class_to_idx}  val={val_ds.class_to_idx}  test={test_ds.class_to_idx}"
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    flags = []
    if dog:     flags.append("DoG: ON")
    if augment: flags.append("augment: ON")
    flag_str = f"  [{', '.join(flags)}]" if flags else ""

    print(
        f"Loaded: {data_dir}\n"
        f"  train={len(train_ds):>6}  val={len(val_ds):>6}  test={len(test_ds):>6}"
        f"  |  class_to_idx={train_ds.class_to_idx}  num_workers={nw}{flag_str}"
    )
    return train_loader, val_loader, test_loader


def get_test_loader(
    data_dir: str,
    batch_size: int = 32,
    already_resized: bool = True,
    num_workers: int | None = None,
    test_split: str = "test",
    dog: bool = False,
) -> DataLoader:
    """
    Convenience function that returns only the test DataLoader.
    Used by evaluate scripts and cross_dataset_matrix.py.

    No augmentation is ever applied to test data — only DoG if specified.

    Parameters
    ----------
    test_split : name of the subfolder to use as the test set.
                 Defaults to 'test'. Pass 'test_balanced' for the
                 balanced test split of dataset_b_balanced.
    dog        : apply Difference of Gaussians preprocessing.
                 Must match the flag used during training.
    """
    _, eval_tf = _build_transforms(already_resized, dog=dog, augment=False)
    nw = num_workers if num_workers is not None else _safe_num_workers()
    test_ds = datasets.ImageFolder(os.path.join(data_dir, test_split), transform=eval_tf)
    dog_str = "  [DoG: ON]" if dog else ""
    print(f"  Test set: {data_dir}/{test_split}  ({len(test_ds)} images){dog_str}")
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )


# ── Per-generator evaluation ───────────────────────────────────────────────────

class GeneratorSubsetDataset(torch.utils.data.Dataset):
    """
    A lightweight Dataset built from an explicit list of (path, label) pairs.
    Used for per-generator evaluation of Dataset B test splits.

    Bypasses ImageFolder — no filesystem scanning required. Reads directly
    from the Dataset B metadata CSV, filtered by label_b (generator id).

    Labels follow the same convention as ImageFolder alphabetical ordering:
        ai=0, real=1
    """

    def __init__(self, samples: list[tuple[str, int]], transform=None):
        """
        Parameters
        ----------
        samples   : list of (absolute_image_path, label) tuples
                    label: 0=ai, 1=real  (matches ImageFolder convention)
        transform : torchvision transform to apply to each image
        """
        self.samples   = samples
        self.transform = transform
        self.class_to_idx = {"ai": 0, "real": 1}
        self.targets   = [label for _, label in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_generator_test_loader(
    dataset_b_dir: str,
    metadata_csv: str,
    generator_id: int,
    batch_size: int = 128,
    already_resized: bool = True,
    num_workers: int | None = None,
    dog: bool = False,
) -> tuple["torch.utils.data.DataLoader", int, int]:
    """
    Build a test DataLoader for a single Dataset B generator subset.

    Includes all real images from the test split paired with all AI images
    from the specified generator (label_b == generator_id).

    Parameters
    ----------
    dataset_b_dir : path to the dataset_b root directory on Drive
    metadata_csv  : path to dataset_b/metadata.csv
    generator_id  : label_b value (1=SD2.1, 2=SDXL, 3=SD3, 4=DALL-E3, 5=MidJourney_v6)
    already_resized : images are pre-resized to 224x224 on disk
    dog           : apply Difference of Gaussians preprocessing

    Returns
    -------
    (loader, n_real, n_ai) — DataLoader and counts for logging
    """
    import pandas as pd

    GENERATOR_NAMES = {
        1: "SD2.1",
        2: "SDXL",
        3: "SD3",
        4: "DALL-E3",
        5: "MidJourney_v6",
    }

    meta = pd.read_csv(metadata_csv)
    test_meta = meta[meta["split"] == "test"]

    # Real images — all real from test split (label_a=0)
    real_records = test_meta[test_meta["label_a"] == 0]
    # AI images — only from the specified generator (label_a=1, label_b=generator_id)
    ai_records   = test_meta[(test_meta["label_a"] == 1) & (test_meta["label_b"] == generator_id)]

    import os
    samples = []
    for _, row in real_records.iterrows():
        path = os.path.join(dataset_b_dir, "test", "real", row["filename"])
        if os.path.exists(path):
            samples.append((path, 1))  # real=1

    for _, row in ai_records.iterrows():
        path = os.path.join(dataset_b_dir, "test", "ai", row["filename"])
        if os.path.exists(path):
            samples.append((path, 0))  # ai=0

    n_real = sum(1 for _, l in samples if l == 1)
    n_ai   = sum(1 for _, l in samples if l == 0)

    gen_name = GENERATOR_NAMES.get(generator_id, f"generator_{generator_id}")
    print(f"  Generator subset: {gen_name}  |  real={n_real}  AI={n_ai}  total={len(samples)}")

    _, eval_tf = _build_transforms(already_resized, dog=dog, augment=False)
    nw = num_workers if num_workers is not None else _safe_num_workers()

    dataset = GeneratorSubsetDataset(samples, transform=eval_tf)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    return loader, n_real, n_ai