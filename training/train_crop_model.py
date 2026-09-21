"""Trains lightweight Crop Identification Model (MobileNetV3) for AgroNerve.

Identifies crop type (Paddy, Cotton, Tomato, Wheat, Chilli, Potato, etc.) from leaf images.
Saves:
- models/crop_model.pth
- models/crop_model.onnx
- models/crop_classes.json
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

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
MODELS_DIR = PROJECT_ROOT / "models"
VISION_DATASET_DIR = DATA_DIR / "vision_dataset"


class CropFolderDataset(Dataset):
    """Loads images organized in class folders and groups them by crop name."""

    def __init__(self, root_dir: Path, crop_to_idx: Dict[str, int], transform=None):
        self.root_dir = root_dir
        self.crop_to_idx = crop_to_idx
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        if not root_dir.exists():
            return

        for class_folder in sorted(root_dir.iterdir()):
            if class_folder.is_dir():
                class_name = class_folder.name
                crop_name = class_name.split("_")[0].lower()
                if crop_name in self.crop_to_idx:
                    label = self.crop_to_idx[crop_name]
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


def build_crop_model(num_classes: int) -> nn.Module:
    """Builds a lightweight MobileNetV3-Small classifier."""
    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    except Exception as e:
        logger.warning(f"Could not load pre-trained MobileNetV3 weights: {e}. Building standalone model.")
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def train_crop_classifier(
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    device_name: str = "auto",
) -> Dict[str, Any]:
    """Trains the crop identification model and exports ONNX + class definitions."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_dir = VISION_DATASET_DIR / "metadata"
    crop_names_path = metadata_dir / "crop_names.json"

    if crop_names_path.exists():
        with open(crop_names_path, "r", encoding="utf-8") as f:
            crop_meta = json.load(f)
            crops = crop_meta["crops"]
            crop_to_idx = crop_meta["crop_to_idx"]
    else:
        # Fallback discovery from folders
        train_dir = VISION_DATASET_DIR / "train"
        crops = sorted(list(set(f.name.split("_")[0].lower() for f in train_dir.iterdir() if f.is_dir())))
        crop_to_idx = {c: i for i, c in enumerate(crops)}

    logger.info(f"Target Crop Classes ({len(crops)}): {crops}")
    if not crops:
        raise ValueError("No crop classes found. Please run training/prepare_dataset.py first.")

    # Datasets
    train_dataset = CropFolderDataset(
        VISION_DATASET_DIR / "train",
        crop_to_idx=crop_to_idx,
        transform=get_train_transforms(),
    )
    val_dataset = CropFolderDataset(
        VISION_DATASET_DIR / "validation",
        crop_to_idx=crop_to_idx,
        transform=get_eval_transforms(),
    )

    logger.info(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Please run training/download_dataset.py and prepare_dataset.py.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if len(val_dataset) > 0 else None

    # Device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    logger.info(f"Training on device: {device}")

    model = build_crop_model(len(crops)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_weights_path = MODELS_DIR / "crop_model.pth"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
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
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
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
            f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        if val_acc >= best_val_acc or epoch == 1:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)
            logger.info(f"Saved best model checkpoint to {best_weights_path} (Val Acc: {val_acc:.2f}%)")

    total_time = round(time.time() - start_time, 2)
    logger.info(f"Crop model training completed in {total_time}s. Best Val Acc: {best_val_acc:.2f}%")

    # Save class definition JSON
    crop_classes_out = {
        "crops": crops,
        "crop_to_idx": crop_to_idx,
        "idx_to_crop": {str(i): c for i, c in enumerate(crops)},
        "total_crops": len(crops),
        "architecture": "mobilenet_v3_small",
        "best_val_acc": best_val_acc,
        "training_time_seconds": total_time,
    }
    with open(MODELS_DIR / "crop_classes.json", "w", encoding="utf-8") as f:
        json.dump(crop_classes_out, f, indent=2)

    # Export ONNX model for high-performance offline inference
    try:
        if sys.platform == "win32":
            try:
                sys.stdout.reconfigure(encoding="utf-8")
                sys.stderr.reconfigure(encoding="utf-8")
            except Exception:
                pass

        model.eval()
        dummy_input = torch.randn(1, 3, settings.VISION_IMAGE_SIZE, settings.VISION_IMAGE_SIZE, device=device)
        onnx_path = MODELS_DIR / "crop_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            dynamo=False,
        )
        logger.info(f"Exported crop model to ONNX: {onnx_path}")
    except Exception as e:
        logger.warning(f"Failed to export crop model to ONNX: {e}")

    return {
        "status": "success",
        "best_val_acc": best_val_acc,
        "history": history,
        "weights_path": str(best_weights_path),
        "crops": crops,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Crop Classifier for AgroNerve.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_crop_classifier(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
