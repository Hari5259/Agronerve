"""Comprehensive evaluation suite for AgroNerve Crop & Disease Vision Models.

Evaluates models strictly on the held-out test split.
Reports:
- Accuracy, Precision, Recall, F1-score (macro and weighted)
- Per-class performance breakdown
- Confusion matrix
- Underperforming class detection

Outputs:
- reports/evaluation_report.json
- reports/classification_report.txt
- reports/confusion_matrix.json
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

from config import settings
from training.transforms import get_eval_transforms
from training.train_crop_model import CropFolderDataset, build_crop_model
from training.train_disease_model import DiseaseFolderDataset, build_disease_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
VISION_DATASET_DIR = DATA_DIR / "vision_dataset"
MODELS_DIR = PROJECT_ROOT / "models"
DISEASE_MODELS_DIR = MODELS_DIR / "disease"
REPORTS_DIR = PROJECT_ROOT / "reports"


def evaluate_crop_model(device: torch.device) -> Dict[str, Any]:
    """Evaluates crop classification model on the test split."""
    crop_classes_file = MODELS_DIR / "crop_classes.json"
    weights_path = MODELS_DIR / "crop_model.pth"

    if not crop_classes_file.exists() or not weights_path.exists():
        logger.warning("Crop model or crop classes file not found. Skipping crop model evaluation.")
        return {"status": "skipped", "reason": "model_not_trained"}

    with open(crop_classes_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
        crops = meta["crops"]
        crop_to_idx = meta["crop_to_idx"]

    test_dataset = CropFolderDataset(
        VISION_DATASET_DIR / "test",
        crop_to_idx=crop_to_idx,
        transform=get_eval_transforms(),
    )

    if len(test_dataset) == 0:
        logger.warning("No test samples found for crop model evaluation.")
        return {"status": "skipped", "reason": "no_test_samples"}

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = build_crop_model(len(crops)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=crops,
        output_dict=True,
        zero_division=0,
    )
    clf_report_text = classification_report(
        y_true,
        y_pred,
        target_names=crops,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    underperforming = [c for c in crops if clf_report.get(c, {}).get("f1-score", 0.0) < 0.70]

    return {
        "status": "success",
        "model_type": "crop_classifier",
        "total_test_samples": len(test_dataset),
        "accuracy_pct": round(acc * 100.0, 2),
        "macro_precision": round(p_macro, 4),
        "macro_recall": round(r_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weighted, 4),
        "per_class_report": clf_report,
        "classification_report_text": clf_report_text,
        "confusion_matrix": cm,
        "underperforming_classes": underperforming,
    }


def evaluate_disease_model(device: torch.device) -> Dict[str, Any]:
    """Evaluates unified disease classification model on the test split."""
    classes_file = DISEASE_MODELS_DIR / "disease_classes.json"
    weights_path = DISEASE_MODELS_DIR / "unified_disease_model.pth"

    if not classes_file.exists() or not weights_path.exists():
        logger.warning("Unified disease model or class definition not found. Skipping.")
        return {"status": "skipped", "reason": "model_not_trained"}

    with open(classes_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
        classes = meta["classes"]
        class_to_idx = meta["class_to_idx"]

    test_dataset = DiseaseFolderDataset(
        VISION_DATASET_DIR / "test",
        class_to_idx=class_to_idx,
        crop_filter=None,
        transform=get_eval_transforms(),
    )

    if len(test_dataset) == 0:
        logger.warning("No test samples found for disease model evaluation.")
        return {"status": "skipped", "reason": "no_test_samples"}

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = build_disease_model(len(classes)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Filter classes present in ground truth or predictions to prevent shape mismatch
    present_indices = sorted(list(set(y_true) | set(y_pred)))
    target_names = [classes[i] for i in present_indices]

    clf_report = classification_report(
        y_true,
        y_pred,
        labels=present_indices,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    clf_report_text = classification_report(
        y_true,
        y_pred,
        labels=present_indices,
        target_names=target_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist()

    underperforming = [c for c in target_names if clf_report.get(c, {}).get("f1-score", 0.0) < 0.70]

    return {
        "status": "success",
        "model_type": "unified_disease_classifier",
        "total_test_samples": len(test_dataset),
        "accuracy_pct": round(acc * 100.0, 2),
        "macro_precision": round(p_macro, 4),
        "macro_recall": round(r_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weighted, 4),
        "per_class_report": clf_report,
        "classification_report_text": clf_report_text,
        "confusion_matrix": cm,
        "underperforming_classes": underperforming,
    }


def run_full_evaluation() -> Dict[str, Any]:
    """Executes evaluation across all models and generates report files."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running model evaluation on device: {device}")

    crop_eval = evaluate_crop_model(device)
    disease_eval = evaluate_disease_model(device)

    overall_report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "crop_model": crop_eval,
        "disease_model": disease_eval,
    }

    # 1. Save JSON evaluation report
    json_path = REPORTS_DIR / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall_report, f, indent=2)

    # 2. Save Human-readable Text report
    txt_path = REPORTS_DIR / "classification_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=====================================================\n")
        f.write("      AGRONERVE VISION PIPELINE EVALUATION REPORT     \n")
        f.write(f"      Generated on: {overall_report['evaluation_timestamp']}\n")
        f.write("=====================================================\n\n")

        f.write("--- 1. CROP IDENTIFICATION MODEL ---\n")
        if crop_eval.get("status") == "success":
            f.write(f"Test Accuracy: {crop_eval['accuracy_pct']}%\n")
            f.write(f"Macro F1-Score: {crop_eval['macro_f1']}\n")
            f.write(f"Weighted F1-Score: {crop_eval['weighted_f1']}\n\n")
            f.write("Per-Class Classification Report:\n")
            f.write(crop_eval.get("classification_report_text", "") + "\n")
            if crop_eval.get("underperforming_classes"):
                f.write(f"⚠️ Underperforming Classes (F1 < 0.70): {crop_eval['underperforming_classes']}\n")
        else:
            f.write(f"Status: {crop_eval.get('status')} ({crop_eval.get('reason')})\n\n")

        f.write("\n--- 2. UNIFIED DISEASE CLASSIFIER MODEL ---\n")
        if disease_eval.get("status") == "success":
            f.write(f"Test Accuracy: {disease_eval['accuracy_pct']}%\n")
            f.write(f"Macro F1-Score: {disease_eval['macro_f1']}\n")
            f.write(f"Weighted F1-Score: {disease_eval['weighted_f1']}\n\n")
            f.write("Per-Class Classification Report:\n")
            f.write(disease_eval.get("classification_report_text", "") + "\n")
            if disease_eval.get("underperforming_classes"):
                f.write(f"⚠️ Underperforming Classes (F1 < 0.70): {disease_eval['underperforming_classes']}\n")
        else:
            f.write(f"Status: {disease_eval.get('status')} ({disease_eval.get('reason')})\n\n")

    # 3. Save Confusion Matrices
    cm_path = REPORTS_DIR / "confusion_matrix.json"
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "crop_confusion_matrix": crop_eval.get("confusion_matrix", []),
                "disease_confusion_matrix": disease_eval.get("confusion_matrix", []),
            },
            f,
            indent=2,
        )

    logger.info(f"Saved evaluation reports to {REPORTS_DIR}")
    return overall_report


if __name__ == "__main__":
    run_full_evaluation()
