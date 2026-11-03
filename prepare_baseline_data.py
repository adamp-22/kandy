"""Audit CIFAKE and create an immutable, duplicate-aware train/validation/test split."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from baseline import LABELS, file_sha256, write_json


def assign_splits(records, validation_per_class=5000, seed=1):
    """Keep the official test set. Deduplicate development data before sampling."""
    labels_by_hash = defaultdict(set)
    for row in records:
        labels_by_hash[row["pixel_sha256"]].add(row["label"])
    if any(len(labels) != 1 for labels in labels_by_hash.values()):
        raise ValueError("Identical pixels have conflicting labels; resolve before training.")
    test_hashes = {r["pixel_sha256"] for r in records if r["source_split"] == "test"}
    seen = set()
    for row in records:
        row["exclusion_reason"] = ""
        if row["source_split"] == "test":
            row["split"] = "test"
        elif row["pixel_sha256"] in test_hashes:
            row["split"], row["exclusion_reason"] = "excluded", "identical_to_official_test"
        elif row["pixel_sha256"] in seen:
            row["split"], row["exclusion_reason"] = "excluded", "repeated_training_image"
        else:
            row["split"] = "train"
            seen.add(row["pixel_sha256"])
    rng = np.random.default_rng(seed)
    for label in LABELS.values():
        eligible = [i for i, r in enumerate(records) if r["split"] == "train" and r["label"] == label]
        if len(eligible) <= validation_per_class:
            raise ValueError("Not enough unique images for the requested validation size")
        for idx in rng.permutation(eligible)[:validation_per_class]:
            records[idx]["split"] = "val"
    hashes = {s: {r["pixel_sha256"] for r in records if r["split"] == s} for s in ("train", "val", "test")}
    if hashes["train"] & hashes["val"] or hashes["train"] & hashes["test"] or hashes["val"] & hashes["test"]:
        raise AssertionError("Pixel overlap across splits")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/cifake"))
    parser.add_argument("--output", type=Path, default=Path("data/baseline_seed1"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--validation-per-class", type=int, default=5000)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}; reuse it or choose a new --output.")
    if args.validation_per_class < 1:
        raise SystemExit("Validation size must be positive")

    records, images = [], []
    for source in ("train", "test"):
        for class_name, label in LABELS.items():
            paths = sorted((args.data_root / source / class_name).glob("*.jpg"))
            expected = 50000 if source == "train" else 10000
            if len(paths) != expected:
                raise ValueError(f"{source}/{class_name}: expected {expected}, found {len(paths)}")
            for path in paths:
                with Image.open(path) as im:
                    if im.size != (32, 32) or im.mode != "RGB":
                        raise ValueError(f"Expected 32x32 RGB: {path}")
                    pixels = np.array(im, dtype=np.uint8)
                images.append(pixels.transpose(2, 0, 1))
                records.append({"path": path.relative_to(args.data_root).as_posix(),
                                "source_split": source, "class_name": class_name, "label": label,
                                "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
            print(f"Decoded and hashed {source}/{class_name}: {len(paths):,}", flush=True)
    assign_splits(records, args.validation_per_class, args.seed)
    args.output.mkdir(parents=True)
    fields = ["path", "source_split", "class_name", "label", "pixel_sha256", "split", "exclusion_reason"]
    with (args.output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    for split in ("train", "val", "test"):
        indices = [i for i, r in enumerate(records) if r["split"] == split]
        np.save(args.output / f"{split}_images.npy", np.stack([images[i] for i in indices]))
        np.save(args.output / f"{split}_labels.npy", np.array([records[i]["label"] for i in indices], dtype=np.int64))
    counts = {s: dict(Counter(r["class_name"] for r in records if r["split"] == s)) for s in ("train", "val", "test", "excluded")}
    test_hashes = [r["pixel_sha256"] for r in records if r["split"] == "test"]
    all_hashes = Counter(r["pixel_sha256"] for r in records)
    audit = {
        "seed": args.seed, "validation_per_class": args.validation_per_class,
        "source_root": str(args.data_root.resolve()), "class_to_index": LABELS,
        "split_counts": counts,
        "exclusion_counts": dict(Counter(r["exclusion_reason"] for r in records if r["split"] == "excluded")),
        "duplicate_groups_in_original_data": sum(v > 1 for v in all_hashes.values()),
        "official_test_size": len(test_hashes), "unique_test_images": len(set(test_hashes)),
        "exact_pixel_overlap_across_splits": 0,
        "policy": "Preserve official test; exclude training copies of test images; retain one copy of each remaining training image; seeded balanced validation holdout.",
        "limitations": "Exact decoded RGB duplicates only; near-duplicates are not detected. Stratified by REAL/FAKE, not unverified semantic classes.",
        "preprocessing": "Decode RGB at native 32x32; float32 / 255 at batch time; no augmentation.",
        "file_sha256": {p.name: file_sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()},
    }
    write_json(args.output / "audit.json", audit)
    print(f"Split counts: {counts}", flush=True)
    print(f"Exclusions: {audit['exclusion_counts']}; unique test images: {len(set(test_hashes)):,}", flush=True)
    print(f"Prepared data: {args.output}", flush=True)


if __name__ == "__main__":
    main()
