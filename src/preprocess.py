"""Data loading & episodic sampling pipeline.
Always respects .cache/ directory for dataset storage. Falls back to synthetic
images if the requested dataset is not present on disk, ensuring the
experiment is entirely self-contained.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

__all__ = ["get_episode_loader"]

# -----------------------------------------------------------------------------
# Visual transforms ------------------------------------------------------------
# -----------------------------------------------------------------------------

def _default_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# -----------------------------------------------------------------------------
# Real miniImageNet wrapper ----------------------------------------------------
# -----------------------------------------------------------------------------

class MiniImageNet(ImageFolder):
    """Extension of ImageFolder with pre-computed class indices for fast sampling."""

    def __init__(self, root: Path):
        super().__init__(root, transform=_default_transform())
        self._cls_to_indices: List[List[int]] = [[] for _ in range(len(self.classes))]
        for idx, (_, cls) in enumerate(self.samples):
            self._cls_to_indices[cls].append(idx)
        assert all(
            self._cls_to_indices
        ), "Dataset split contains empty classes – check integrity."

    @property
    def class_to_indices(self) -> List[List[int]]:  # noqa: D401
        return self._cls_to_indices


# -----------------------------------------------------------------------------
# Synthetic fallback -----------------------------------------------------------
# -----------------------------------------------------------------------------

class SyntheticFewShotDataset(Dataset):
    """Generates random RGB images on-the-fly with deterministic class labels."""

    def __init__(self, num_classes: int, images_per_class: int = 120):
        self.num_classes = num_classes
        self.images_per_class = images_per_class
        self._len = num_classes * images_per_class
        self._cls_to_indices: List[List[int]] = [
            list(range(c * images_per_class, (c + 1) * images_per_class))
            for c in range(num_classes)
        ]
        self._norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:  # noqa: D401
        return self._len

    def __getitem__(self, idx: int):  # noqa: D401, ANN201
        cls = idx // self.images_per_class
        img = torch.rand(3, 224, 224)
        img = self._norm(img)
        return img, cls

    @property
    def class_to_indices(self) -> List[List[int]]:  # noqa: D401
        return self._cls_to_indices


# -----------------------------------------------------------------------------
# Episode sampling -------------------------------------------------------------
# -----------------------------------------------------------------------------

def _sample_episode(
    ds, n_ways: int, n_shots: int, n_queries: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (support_imgs, support_lbls, query_imgs, query_lbls)."""

    chosen_classes = random.sample(range(len(ds.class_to_indices)), n_ways)

    sup_imgs, sup_lbls, qry_imgs, qry_lbls = [], [], [], []
    for new_lbl, cls in enumerate(chosen_classes):
        pool = ds.class_to_indices[cls]
        assert len(pool) >= n_shots + n_queries, "Class lacks enough images"
        selected = random.sample(pool, n_shots + n_queries)
        sup_ids, qry_ids = selected[:n_shots], selected[n_shots:]
        for idx in sup_ids:
            img, _ = ds[idx]
            sup_imgs.append(img)
            sup_lbls.append(new_lbl)
        for idx in qry_ids:
            img, _ = ds[idx]
            qry_imgs.append(img)
            qry_lbls.append(new_lbl)

    return (
        torch.stack(sup_imgs),
        torch.tensor(sup_lbls, dtype=torch.long),
        torch.stack(qry_imgs),
        torch.tensor(qry_lbls, dtype=torch.long),
    )


class EpisodeIterator(Iterator):
    """Simple iterator yielding episodic batches according to configuration."""

    def __init__(
        self,
        ds,
        n_episodes: int,
        n_ways: int,
        n_shots: int,
        n_queries: int,
        *,
        seed: int = 0,
    ):
        self.ds = ds
        self.n_episodes = n_episodes
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self._idx = 0
        self._saved_state = random.getstate()
        random.seed(seed)

    def __iter__(self):
        return self

    def __next__(self):  # noqa: D401
        if self._idx >= self.n_episodes:
            random.setstate(self._saved_state)
            raise StopIteration
        self._idx += 1
        return _sample_episode(
            self.ds, self.n_ways, self.n_shots, self.n_queries
        )


# -----------------------------------------------------------------------------
# Factory ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def get_episode_loader(dataset_cfg, *, seed: int = 0):  # noqa: ANN001, ANN201
    """Return an EpisodeIterator prepared from `dataset_cfg`. Always uses .cache/."""

    if dataset_cfg.name.lower() != "miniimagenet":
        raise NotImplementedError("Only the miniImageNet pipeline is implemented.")

    default_root = Path(".cache/datasets") / dataset_cfg.name
    root = Path(getattr(dataset_cfg, "data_root", default_root)).expanduser().resolve()

    has_images = root.exists() and any(root.rglob("*.jpg"))
    if has_images:
        print(f"[Preprocess] Loading real miniImageNet images from {root}")
        ds = MiniImageNet(root)
    else:
        print(f"[Preprocess] No images found at {root}. Falling back to synthetic data.")
        n_classes_syn = max(int(dataset_cfg.n_ways) * 2, 20)
        images_per_cls = max(
            int(dataset_cfg.n_shots[0] if isinstance(dataset_cfg.n_shots, (list, tuple)) else dataset_cfg.n_shots)
            + int(dataset_cfg.n_queries),
            120,
        )
        ds = SyntheticFewShotDataset(
            num_classes=n_classes_syn, images_per_class=images_per_cls
        )

    first_n_shots = (
        dataset_cfg.n_shots[0]
        if isinstance(dataset_cfg.n_shots, (list, tuple))
        else dataset_cfg.n_shots
    )

    return EpisodeIterator(
        ds=ds,
        n_episodes=int(dataset_cfg.n_episodes),
        n_ways=int(dataset_cfg.n_ways),
        n_shots=int(first_n_shots),
        n_queries=int(dataset_cfg.n_queries),
        seed=seed,
    )
