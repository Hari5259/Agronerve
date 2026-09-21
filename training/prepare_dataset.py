"""Prepares train/validation/test splits and metadata for AgroNerve Vision Datasets.

Splits:
- Train: ~70%
- Validation: ~15%
- Test: ~15%

Outputs:
- data/vision_dataset/train/<class_name>/
- data/vision_dataset/validation/<class_name>/
- data/vision_dataset/test/<class_name>/
- data/vision_dataset/metadata/dataset.csv
- data/vision_dataset/metadata/class_names.json
- data/vision_dataset/metadata/crop_names.json
"""

import os
import sys
import shutil
import random
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.download_dataset import load_target_classes_from_kb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
VISION_DATASET_DIR = DATA_DIR / "vision_dataset"
RAW_STAGING_DIR = VISION_DATASET_DIR / "raw"
METADATA_DIR = VISION_DATASET_DIR / "metadata"


def prepare_dataset_splits(
    raw_dir: Path = RAW_STAGING_DIR,
    output_base_dir: Path = VISION_DATASET_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Splits raw images into train, validation, and test sets and generates dataset metadata."""
    random.seed(seed)
    target_classes = load_target_classes_from_kb()

    train_dir = output_base_dir / "train"
    val_dir = output_base_dir / "validation"
    test_dir = output_base_dir / "test"
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    class_split_counts: Dict[str, Dict[str, int]] = {}
    total_images_processed = 0

    if not raw_dir.exists():
        logger.warning(f"Raw directory {raw_dir} does not exist.")
        return {"status": "empty", "total_images": 0}

    class_folders = [f for f in raw_dir.iterdir() if f.is_dir()]
    logger.info(f"Found {len(class_folders)} class directories in raw staging.")

    for class_folder in sorted(class_folders):
        class_name = class_folder.name
        class_info = target_classes.get(
            class_name,
            {
                "crop": class_name.split("_")[0],
                "crop_display": class_name.split("_")[0].capitalize(),
                "disease_name": class_name.replace(class_name.split("_")[0] + "_", "").replace("_", " ").title(),
                "disease_id": f"DIS_{class_name.upper()}",
            },
        )

        image_files = sorted([f for f in class_folder.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
        if not image_files:
            continue

        # Deterministic shuffle
        random.shuffle(image_files)
        n_total = len(image_files)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        # Ensure at least 1 per split if n_total >= 3
        if n_total >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = n_total - n_train - n_val
            if n_test <= 0:
                n_train -= 1
                n_test = 1
        elif n_total == 2:
            n_train, n_val, n_test = 1, 0, 1
        else:
            n_train, n_val, n_test = 1, 0, 0

        train_files = image_files[:n_train]
        val_files = image_files[n_train : n_train + n_val]
        test_files = image_files[n_train + n_val :]

        split_map = {
            "train": (train_files, train_dir),
            "validation": (val_files, val_dir),
            "test": (test_files, test_dir),
        }

        class_split_counts[class_name] = {
            "train": len(train_files),
            "validation": len(val_files),
            "test": len(test_files),
            "total": n_total,
        }

        for split_name, (files, dest_root) in split_map.items():
            dest_class_dir = dest_root / class_name
            dest_class_dir.mkdir(parents=True, exist_ok=True)

            for f in files:
                dest_file = dest_class_dir / f.name
                shutil.copy2(f, dest_file)
                total_images_processed += 1

                image_id = f.stem
                rel_path = str(dest_file.relative_to(PROJECT_ROOT)).replace("\\", "/")

                records.append(
                    {
                        "image_id": image_id,
                        "image_path": rel_path,
                        "crop_name": class_info["crop_display"],
                        "crop_id": class_info["crop"],
                        "disease_name": class_info["disease_name"],
                        "disease_id": class_info.get("disease_id", "N/A"),
                        "class_name": class_name,
                        "source": "AgroNerve_Vision_Corpus",
                        "split": split_name,
                    }
                )

    # 1. Generate dataset.csv
    df = pd.DataFrame(records)
    csv_path = METADATA_DIR / "dataset.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved dataset metadata CSV to {csv_path} with {len(df)} entries.")

    # 2. Generate class_names.json
    unique_classes = sorted(list(set(r["class_name"] for r in records)))
    class_names_json_path = METADATA_DIR / "class_names.json"
    with open(class_names_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": unique_classes,
                "class_to_idx": {c: i for i, c in enumerate(unique_classes)},
                "idx_to_class": {i: c for i, c in enumerate(unique_classes)},
                "total_classes": len(unique_classes),
            },
            f,
            indent=2,
        )

    # 3. Generate crop_names.json
    unique_crops = sorted(list(set(r["crop_id"] for r in records)))
    crop_names_json_path = METADATA_DIR / "crop_names.json"
    with open(crop_names_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "crops": unique_crops,
                "crop_to_idx": {c: i for i, c in enumerate(unique_crops)},
                "idx_to_crop": {i: c for i, c in enumerate(unique_crops)},
                "total_crops": len(unique_crops),
            },
            f,
            indent=2,
        )

    summary = {
        "status": "success",
        "total_images": total_images_processed,
        "classes_count": len(unique_classes),
        "crops_count": len(unique_crops),
        "split_breakdown": {
            "train": sum(c["train"] for c in class_split_counts.values()),
            "validation": sum(c["validation"] for c in class_split_counts.values()),
            "test": sum(c["test"] for c in class_split_counts.values()),
        },
        "per_class_counts": class_split_counts,
    }

    with open(METADATA_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Dataset preparation complete! Total: {total_images_processed} images.")
    logger.info(f"Splits: Train={summary['split_breakdown']['train']}, Val={summary['split_breakdown']['validation']}, Test={summary['split_breakdown']['test']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare train/val/test splits for AgroNerve vision dataset.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset_splits(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
