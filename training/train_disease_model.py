"""Trains Modular Disease Classification Models (MobileNetV3) for AgroNerve.

Supports:
1. Unified Multi-Class Disease Classifier (models/disease/unified_disease_model.pth)
2. Crop-Specific Disease Classifiers (models/disease/<crop>_disease_model.pth)

Exports:
- PyTorch state_dicts (.pth)
- High-performance ONNX models (.onnx)
- Metadata & Class mapping JSONs (models/disease/disease_classes.json)
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torchvision.models as models

from config import settings
from training.transforms import get_train_transforms, get_eval_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
VISION_DATASET_DIR = DATA_DIR / "vision_dataset"
DISEASE_MODELS_DIR = PROJECT_ROOT / "models" / "disease"


class DiseaseFolderDataset(Dataset):
    """Loads images organized in canonical class folders with optional crop filtering."""

    def __init__(
        self,
        root_dir: Path,
        class_to_idx: Dict[str, int],
        crop_filter: Optional[str] = None,
        transform=None,
    ):
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.crop_filter = crop_filter.lower() if crop_filter else None
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        if not root_dir.exists():
            return

        for class_folder in sorted(root_dir.iterdir()):
            if class_folder.is_dir():
                class_name = class_folder.name
                if self.crop_filter:
                    class_crop = class_name.split("_")[0].lower()
                    if class_crop != self.crop_filter:
                        continue

                if class_name in self.class_to_idx:
                    label = self.class_to_idx[class_name]
                    for img_p in class_folder.glob("*.*"):
                        if img_p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                            self.samples.append((img_p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_disease_model(num_classes: int) -> nn.Module:
    """Builds a lightweight MobileNetV3-Small classifier for disease diagnosis."""
    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    except Exception as e:
        logger.warning(f"Could not load pre-trained weights: {e}. Building standalone model.")
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def train_single_model(
    model_name: str,
    classes: List[str],
    class_to_idx: Dict[str, int],
    crop_filter: Optional[str] = None,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Trains a disease classification model for a given set of classes."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"--- Training {model_name} ({len(classes)} classes) on {device} ---")
    logger.info(f"Classes: {classes}")

    train_dataset = DiseaseFolderDataset(
        VISION_DATASET_DIR / "train",
        class_to_idx=class_to_idx,
        crop_filter=crop_filter,
        transform=get_train_transforms(),
    )
    val_dataset = DiseaseFolderDataset(
        VISION_DATASET_DIR / "validation",
        class_to_idx=class_to_idx,
        crop_filter=crop_filter,
        transform=get_eval_transforms(),
    )

    logger.info(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images for {model_name}.")
    if len(train_dataset) == 0:
        logger.warning(f"No training samples found for {model_name}. Skipping.")
        return {"status": "skipped", "reason": "empty_dataset"}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if len(val_dataset) > 0 else None

    model = build_disease_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    weights_path = DISEASE_MODELS_DIR / f"{model_name}.pth"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"{model_name} Ep {epoch}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = (correct / max(1, total)) * 100.0
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))

        # Validation
        val_loss, val_acc = 0.0, 0.0
        if val_loader and len(val_dataset) > 0:
            model.eval()
            val_running_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss = val_running_loss / max(1, val_total)
            val_acc = (val_correct / max(1, val_total)) * 100.0

        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 2))

        logger.info(
            f"{model_name} Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        if val_acc >= best_val_acc or epoch == 1:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weights_path)
            logger.info(f"Checkpoint saved: {weights_path} (Val Acc: {val_acc:.2f}%)")

    total_time = round(time.time() - start_time, 2)

    # Export to ONNX
    try:
        if sys.platform == "win32":
            try:
                sys.stdout.reconfigure(encoding="utf-8")
                sys.stderr.reconfigure(encoding="utf-8")
            except Exception:
                pass

        model.eval()
        dummy_input = torch.randn(1, 3, settings.VISION_IMAGE_SIZE, settings.VISION_IMAGE_SIZE, device=device)
        onnx_path = DISEASE_MODELS_DIR / f"{model_name}.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            dynamo=False,
        )
        logger.info(f"Exported ONNX model: {onnx_path}")
    except Exception as e:
        logger.warning(f"Failed to export {model_name} to ONNX: {e}")

    return {
        "status": "success",
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "classes": classes,
        "history": history,
        "training_time_seconds": total_time,
    }


def train_disease_pipeline(
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    train_per_crop: bool = True,
) -> Dict[str, Any]:
    """Trains the unified disease model and optionally per-crop specialist models."""
    DISEASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_dir = VISION_DATASET_DIR / "metadata"
    class_names_path = metadata_dir / "class_names.json"

    if class_names_path.exists():
        with open(class_names_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            all_classes = meta["classes"]
            all_class_to_idx = meta["class_to_idx"]
    else:
        train_dir = VISION_DATASET_DIR / "train"
        all_classes = sorted([f.name for f in train_dir.iterdir() if f.is_dir()])
        all_class_to_idx = {c: i for i, c in enumerate(all_classes)}

    if not all_classes:
        raise ValueError("No classes found. Please run training/prepare_dataset.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    # 1. Train Unified Disease Model (all classes)
    unified_res = train_single_model(
        model_name="unified_disease_model",
        classes=all_classes,
        class_to_idx=all_class_to_idx,
        crop_filter=None,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    results["unified_disease_model"] = unified_res

    # 2. Train Crop-Specific Models where multiple disease classes exist
    if train_per_crop:
        # Group classes by crop
        crop_groups: Dict[str, List[str]] = {}
        for c in all_classes:
            crop = c.split("_")[0].lower()
            crop_groups.setdefault(crop, []).append(c)

        for crop, crop_classes in crop_groups.items():
            if len(crop_classes) >= 2:
                crop_class_to_idx = {c: i for i, c in enumerate(crop_classes)}
                crop_res = train_single_model(
                    model_name=f"{crop}_disease_model",
                    classes=crop_classes,
                    class_to_idx=crop_class_to_idx,
                    crop_filter=crop,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    device=device,
                )
                results[f"{crop}_disease_model"] = crop_res

    # Save comprehensive metadata
    disease_classes_meta = {
        "classes": all_classes,
        "class_to_idx": all_class_to_idx,
        "idx_to_class": {str(i): c for i, c in enumerate(all_classes)},
        "total_classes": len(all_classes),
        "results": results,
    }
    with open(DISEASE_MODELS_DIR / "disease_classes.json", "w", encoding="utf-8") as f:
        json.dump(disease_classes_meta, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Disease Classifier for AgroNerve.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-per-crop", action="store_true", help="Disable training per-crop specialist models")
    args = parser.parse_args()

    train_disease_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_per_crop=not args.no_per_crop,
    )
