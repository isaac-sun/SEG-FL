#!/usr/bin/env python3
"""Build the fixed-size VOC 2012 (+SBD) arrays used by SEG-FL.

Output (``--out``, default ``dataset/voc_seg``)::

    train_images.npy  uint8 [N, S, S, 3]   train_aug pool (clients + reference)
    train_labels.npy  uint8 [N, S, S]      0..20, 255 = ignore
    val_images.npy    uint8 [M, S, S, 3]   official val = test set
    val_labels.npy    uint8 [M, S, S]
    train_ids.txt / val_ids.txt
    meta.json

Every image is resized so its shorter side is S (bilinear; labels nearest)
and center-cropped to S x S (S = 288 by default).  Training takes random
256 x 256 crops of these; evaluation takes the 256 x 256 center crop.

Label sources, in order of preference:

1. ``--aug-labels DIR``: SegmentationClassAug PNGs (one per train_aug image),
   optionally with ``--train-list train_aug.txt``;
2. ``--sbd-root DIR``: the SBD release (``cls/*.mat``, ``train.txt``,
   ``val.txt``) merged with VOC SegmentationClass, the standard recipe:
   train_aug = (SBD train + SBD val + VOC train) - VOC val = 10,582 images;
3. neither: VOC SegmentationClass only (1,464 train images).

``--synthetic N`` writes a small synthetic dataset in the same format
(coloured shapes on noise) for tests and dry runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

VOC_CLASSES = ("background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
               "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
               "tvmonitor")


def read_list(path: str) -> list[str]:
    with open(path) as handle:
        return [line.split()[0].strip() for line in handle if line.strip()]


def resize_center(image, size: int, is_label: bool):
    from PIL import Image

    width, height = image.size
    scale = size / min(width, height)
    new = (max(size, round(width * scale)), max(size, round(height * scale)))
    image = image.resize(new, Image.NEAREST if is_label else Image.BILINEAR)
    left = (new[0] - size) // 2
    top = (new[1] - size) // 2
    return image.crop((left, top, left + size, top + size))


def load_label(image_id: str, sources: dict):
    from PIL import Image

    aug = sources.get("aug_labels")
    if aug:
        path = os.path.join(aug, image_id + ".png")
        if os.path.exists(path):
            return Image.fromarray(np.array(Image.open(path)).astype(np.uint8))
    voc = os.path.join(sources["voc_root"], "SegmentationClass", image_id + ".png")
    if os.path.exists(voc):   # palette PNG: the array holds class indices
        return Image.fromarray(np.array(Image.open(voc)).astype(np.uint8))
    sbd = sources.get("sbd_root")
    if sbd:
        from scipy.io import loadmat

        mat = loadmat(os.path.join(sbd, "cls", image_id + ".mat"))
        return Image.fromarray(mat["GTcls"][0]["Segmentation"][0].astype(np.uint8))
    raise FileNotFoundError(f"no label for {image_id}")


def process(args):
    image_id, sources, size = args
    from PIL import Image

    image = Image.open(os.path.join(sources["voc_root"], "JPEGImages",
                                    image_id + ".jpg")).convert("RGB")
    label = load_label(image_id, sources)
    if label.size != image.size:
        raise ValueError(f"{image_id}: image {image.size} vs label {label.size}")
    image = np.asarray(resize_center(image, size, False), dtype=np.uint8)
    label = np.asarray(resize_center(label, size, True), dtype=np.uint8)
    bad = (label > 20) & (label != 255)
    if bad.any():
        raise ValueError(f"{image_id}: unexpected label values {np.unique(label[bad])}")
    return image, label


def write_split(out: str, name: str, ids: list[str], sources: dict, size: int,
                workers: int) -> dict:
    images = np.lib.format.open_memmap(os.path.join(out, f"{name}_images.npy.tmp"),
                                       mode="w+", dtype=np.uint8,
                                       shape=(len(ids), size, size, 3))
    labels = np.lib.format.open_memmap(os.path.join(out, f"{name}_labels.npy.tmp"),
                                       mode="w+", dtype=np.uint8,
                                       shape=(len(ids), size, size))
    pixel_counts = np.zeros(256, dtype=np.int64)
    jobs = [(image_id, sources, size) for image_id in ids]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (image, label) in enumerate(pool.map(process, jobs, chunksize=32)):
            images[i] = image
            labels[i] = label
            pixel_counts += np.bincount(label.reshape(-1), minlength=256)
            if (i + 1) % 1000 == 0:
                print(f"  {name}: {i + 1}/{len(ids)}", flush=True)
    images.flush()
    labels.flush()
    del images, labels
    for kind in ("images", "labels"):
        os.replace(os.path.join(out, f"{name}_{kind}.npy.tmp"),
                   os.path.join(out, f"{name}_{kind}.npy"))
    with open(os.path.join(out, f"{name}_ids.txt"), "w") as handle:
        handle.write("\n".join(ids) + "\n")
    return {"num_images": len(ids),
            "class_pixels": pixel_counts[:21].tolist(),
            "ignore_pixels": int(pixel_counts[255])}


def train_aug_ids(args, val_ids: list[str]) -> tuple[list[str], str]:
    val = set(val_ids)
    if args.train_list:
        return [i for i in read_list(args.train_list) if i not in val], "train_list"
    if args.aug_labels:
        ids = sorted(f[:-4] for f in os.listdir(args.aug_labels) if f.endswith(".png"))
        return [i for i in ids if i not in val], "aug_labels"
    voc_train = read_list(os.path.join(args.voc_root, "ImageSets", "Segmentation", "train.txt"))
    if args.sbd_root:
        sbd = read_list(os.path.join(args.sbd_root, "train.txt")) + \
            read_list(os.path.join(args.sbd_root, "val.txt"))
        return sorted(set(sbd + voc_train) - val), "sbd"
    return voc_train, "voc_train_only"


# ── synthetic data for tests ──────────────────────────────────────────────

def synthetic_split(num_images: int, size: int, seed: int):
    rng = np.random.default_rng(seed)
    palette = rng.integers(30, 226, size=(21, 3))
    yy, xx = np.mgrid[0:size, 0:size]
    images = np.empty((num_images, size, size, 3), dtype=np.uint8)
    labels = np.zeros((num_images, size, size), dtype=np.uint8)
    for n in range(num_images):
        canvas = palette[0] + rng.normal(0, 25, size=(size, size, 3))
        label = np.zeros((size, size), dtype=np.uint8)
        for _ in range(rng.integers(1, 4)):
            c = int(rng.integers(1, 21))
            r = int(rng.integers(size // 8, size // 3))
            cy, cx = rng.integers(r, size - r, size=2)
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
            if c % 2:   # odd classes are squares
                mask = (np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)
            canvas[mask] = palette[c] + rng.normal(0, 25, size=(int(mask.sum()), 3))
            label[mask] = c
        edge = np.zeros_like(label, dtype=bool)
        edge[1:] |= label[1:] != label[:-1]
        label[edge] = 255
        images[n] = np.clip(canvas, 0, 255).astype(np.uint8)
        labels[n] = label
    return images, labels


def write_synthetic(out: str, num_images: int, size: int) -> None:
    os.makedirs(out, exist_ok=True)
    meta = {"source": "synthetic", "store_size": size, "classes": list(VOC_CLASSES)}
    for name, count, seed in (("train", num_images, 0), ("val", max(8, num_images // 5), 1)):
        images, labels = synthetic_split(count, size, seed)
        np.save(os.path.join(out, f"{name}_images.npy"), images)
        np.save(os.path.join(out, f"{name}_labels.npy"), labels)
        with open(os.path.join(out, f"{name}_ids.txt"), "w") as handle:
            handle.write("\n".join(f"syn_{name}_{i:05d}" for i in range(count)) + "\n")
        meta[name] = {"num_images": count,
                      "class_pixels": np.bincount(labels.reshape(-1), minlength=256)[:21].tolist()}
    with open(os.path.join(out, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"wrote synthetic data to {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voc-root", help="…/VOCdevkit/VOC2012")
    p.add_argument("--aug-labels", help="SegmentationClassAug directory")
    p.add_argument("--train-list", help="train_aug.txt (optional)")
    p.add_argument("--sbd-root", help="SBD 'dataset' directory (cls/, train.txt, val.txt)")
    p.add_argument("--out", default="dataset/voc_seg")
    p.add_argument("--store-size", type=int, default=288)
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--limit", type=int, default=None, help="first N ids per split (testing)")
    p.add_argument("--synthetic", type=int, default=None, metavar="N")
    args = p.parse_args(argv)

    if args.synthetic:
        write_synthetic(args.out, args.synthetic, args.store_size)
        return 0
    if not args.voc_root:
        p.error("--voc-root is required (or --synthetic N)")
    os.makedirs(args.out, exist_ok=True)
    sources = {"voc_root": args.voc_root, "aug_labels": args.aug_labels,
               "sbd_root": args.sbd_root}
    val_ids = read_list(os.path.join(args.voc_root, "ImageSets", "Segmentation", "val.txt"))
    train_ids, train_source = train_aug_ids(args, val_ids)
    if args.limit:
        train_ids, val_ids = train_ids[:args.limit], val_ids[:args.limit]
    print(f"train_aug: {len(train_ids)} images ({train_source}); val: {len(val_ids)}")
    if train_source == "voc_train_only":
        print("WARNING: no SBD / SegmentationClassAug labels given; only 1,464 training images",
              file=sys.stderr)
    meta = {"source": "voc2012", "train_source": train_source,
            "store_size": args.store_size, "classes": list(VOC_CLASSES)}
    meta["train"] = write_split(args.out, "train", train_ids, sources,
                                args.store_size, args.workers)
    meta["val"] = write_split(args.out, "val", val_ids, sources,
                              args.store_size, args.workers)
    with open(os.path.join(args.out, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"done: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
