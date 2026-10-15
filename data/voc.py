"""VOC 2012 arrays, reference set, augmentation and Shapley groupings.

The arrays come from ``prepare_data.py`` (S x S, S = 288 by default).  The
train_aug pool is split into the server reference set (``reference_per_class``
images per dominant foreground class, never given to clients) and the
client pool.  The official val split is the test set.

"Classes" for the per-class Shapley values are pixel labels.  A grouping maps
every (reference image, pixel class) cell to a group id, so one inference of a
coalition model over the reference set gives the group losses of every
grouping (see ``contribution/shapley.py``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

VOC_CLASSES = ("background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
               "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
               "tvmonitor")
# Official VOC 2012 categories: person, animal, vehicle, indoor.
VOC_SUPER = {
    "background": (0,),
    "person": (15,),
    "animal": (3, 8, 10, 12, 13, 17),
    "vehicle": (1, 2, 4, 6, 7, 14, 19),
    "indoor": (5, 9, 11, 16, 18, 20),
}
GROUPINGS = ("class21", "class20_nobg", "voc_super5", "random_class5", "random_image21")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ── loading ────────────────────────────────────────────────────────────────

def dominant_classes(labels: np.ndarray, num_classes: int = 21) -> np.ndarray:
    """Foreground class with the most pixels per image (0 if none)."""
    out = np.zeros(len(labels), dtype=np.int64)
    for i, label in enumerate(labels):
        counts = np.bincount(label.reshape(-1), minlength=256)[1:num_classes]
        out[i] = int(np.argmax(counts)) + 1 if counts.sum() else 0
    return out


def select_reference(dominant: np.ndarray, per_class: int, seed: int,
                     num_classes: int = 21) -> np.ndarray:
    """``per_class`` images per dominant foreground class (fewer if scarce)."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c in range(1, num_classes):
        members = np.flatnonzero(dominant == c)
        take = min(per_class, len(members))
        chosen.extend(rng.choice(members, size=take, replace=False).tolist())
    return np.array(sorted(chosen), dtype=np.int64)


def center_crop(array: np.ndarray, size: int) -> np.ndarray:
    """Center crop of ``[N, S, S, ...]`` to ``[N, size, size, ...]``."""
    s = array.shape[1]
    top = (s - size) // 2
    return array[:, top:top + size, top:top + size]


@dataclass
class VOCData:
    train_images: torch.Tensor     # uint8 [N, S, S, 3], client pool + reference
    train_labels: torch.Tensor     # uint8 [N, S, S]
    dominant: np.ndarray           # [N] dominant foreground class
    reference_index: np.ndarray    # indices into train_* (held out)
    client_pool: np.ndarray        # indices into train_* given to clients
    test_images: torch.Tensor      # uint8 [M, C, C, 3] center crops
    test_labels: torch.Tensor      # uint8 [M, C, C]
    meta: dict


def load_voc(config) -> VOCData:
    folder = config.data_dir
    path = lambda name: os.path.join(folder, name)  # noqa: E731
    if not os.path.exists(path("train_images.npy")):
        raise FileNotFoundError(
            f"{folder}/train_images.npy not found; run prepare_data.py first")
    train_images = np.load(path("train_images.npy"))
    train_labels = np.load(path("train_labels.npy"))
    test_images = center_crop(np.load(path("val_images.npy")), config.crop_size)
    test_labels = center_crop(np.load(path("val_labels.npy")), config.crop_size)
    meta = {}
    if os.path.exists(path("meta.json")):
        with open(path("meta.json")) as handle:
            meta = json.load(handle)
    if train_images.shape[1] < config.crop_size:
        raise ValueError("stored images are smaller than crop_size")
    dominant = dominant_classes(train_labels, config.num_classes)
    reference = select_reference(dominant, config.reference_per_class,
                                 config.reference_seed, config.num_classes)
    pool = np.setdiff1d(np.arange(len(train_images)), reference)
    return VOCData(
        train_images=torch.from_numpy(np.ascontiguousarray(train_images)),
        train_labels=torch.from_numpy(np.ascontiguousarray(train_labels)),
        dominant=dominant,
        reference_index=reference,
        client_pool=pool,
        test_images=torch.from_numpy(np.ascontiguousarray(test_images)),
        test_labels=torch.from_numpy(np.ascontiguousarray(test_labels)),
        meta=meta,
    )


# ── batches ────────────────────────────────────────────────────────────────

def normalize(images_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 NHWC -> float NCHW with ImageNet normalization (same device)."""
    x = images_uint8.permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def random_crop_flip(images: torch.Tensor, labels: torch.Tensor, size: int,
                     generator: torch.Generator):
    """Random ``size`` crop and horizontal flip per sample (CPU tensors)."""
    n, s = images.shape[0], images.shape[1]
    tops = torch.randint(0, s - size + 1, (n,), generator=generator)
    lefts = torch.randint(0, s - size + 1, (n,), generator=generator)
    flips = torch.rand(n, generator=generator) < 0.5
    out_images = torch.empty((n, size, size, 3), dtype=images.dtype)
    out_labels = torch.empty((n, size, size), dtype=labels.dtype)
    for i in range(n):
        t, l = int(tops[i]), int(lefts[i])
        img = images[i, t:t + size, l:l + size]
        lab = labels[i, t:t + size, l:l + size]
        if flips[i]:
            img, lab = img.flip(1), lab.flip(1)
        out_images[i] = img
        out_labels[i] = lab
    return out_images, out_labels


class ImageSet:
    """Fixed evaluation images (center crops) kept on the evaluation device."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, batch_size: int,
                 device: str, num_classes: int = 21):
        self.images = images.to(device)
        self.labels = labels.to(device).long()
        self.batch_size = batch_size
        self.device = device
        self.num_classes = num_classes

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def batches(self):
        for start in range(0, len(self), self.batch_size):
            stop = min(start + self.batch_size, len(self))
            yield start, normalize(self.images[start:stop]), self.labels[start:stop]

    def pixel_counts(self) -> np.ndarray:
        """``[num_images, num_classes]`` labelled pixels per image and class."""
        counts = torch.zeros(len(self) * self.num_classes, dtype=torch.float64,
                             device=self.labels.device)
        for start, _, y in self.batches():
            idx = torch.arange(start, start + y.shape[0], device=y.device).view(-1, 1, 1)
            valid = y != 255
            flat = (idx * self.num_classes + y)[valid]
            counts += torch.bincount(flat, minlength=counts.numel()).double()
        return counts.view(len(self), self.num_classes).cpu().numpy()


# ── groupings ──────────────────────────────────────────────────────────────

@dataclass
class Grouping:
    name: str
    num_groups: int
    cell_group: np.ndarray        # [num_images, num_classes], -1 = unused cell
    group_names: list
    definition: str


def _class_grouping(name, class_to_group, group_names, num_images, definition):
    cell = np.tile(np.asarray(class_to_group, dtype=np.int64), (num_images, 1))
    return Grouping(name, len(group_names), cell, list(group_names), definition)


def build_groupings(num_images: int, num_classes: int = 21, seed: int = 0) -> dict:
    """The five pre-registered groupings of the reference cells."""
    rng = np.random.default_rng(seed)
    groupings = {}
    groupings["class21"] = _class_grouping(
        "class21", np.arange(num_classes), VOC_CLASSES[:num_classes], num_images,
        "every pixel grouped by its label (main)")
    nobg = np.arange(num_classes) - 1           # background -> -1
    groupings["class20_nobg"] = _class_grouping(
        "class20_nobg", nobg, VOC_CLASSES[1:num_classes], num_images,
        "as class21 without background")
    super_map = np.full(num_classes, -1, dtype=np.int64)
    for g, classes in enumerate(VOC_SUPER.values()):
        for c in classes:
            if c < num_classes:
                super_map[c] = g
    groupings["voc_super5"] = _class_grouping(
        "voc_super5", super_map, list(VOC_SUPER), num_images,
        "official VOC categories: background, person, animal, vehicle, indoor")
    sizes = np.bincount(super_map[super_map >= 0])
    shuffled = rng.permutation(num_classes)
    random_map = np.empty(num_classes, dtype=np.int64)
    start = 0
    for g, size in enumerate(sizes):
        random_map[shuffled[start:start + size]] = g
        start += size
    groupings["random_class5"] = _class_grouping(
        "random_class5", random_map, [f"rand_{g}" for g in range(len(sizes))], num_images,
        "classes merged at random into groups with the voc_super5 sizes")
    image_group = np.arange(num_images) % num_classes
    image_group = image_group[rng.permutation(num_images)]
    cell = np.repeat(image_group[:, None], num_classes, axis=1)
    groupings["random_image21"] = Grouping(
        "random_image21", num_classes, cell, [f"img_{g}" for g in range(num_classes)],
        "reference images split at random into 21 groups; all pixels of an image together")
    return groupings
