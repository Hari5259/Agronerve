# AgroNerve Model Training & Evaluation Guide

This document describes the computer vision training architecture, transfer learning procedures, ONNX export, and evaluation metrics for AgroNerve.

---

## 1. Architectural Overview

AgroNerve utilizes a modular two-stage vision diagnostic pipeline:

```
[Input Image Bytes]
       ↓
[Image Quality Filter] (Blur, Luminance, Resolution, Foliage %)
       ↓
[Crop Identification Model (MobileNetV3)]
       ↓
[Crop Confidence Check] (Threshold >= 0.60)
       ↓
[Disease Classification Model (MobileNetV3)]
       ↓
[Disease Confidence Check] (Threshold >= 0.60)
       ↓
[Grounding Knowledge Base Lookup (disease.json)]
       ↓
[Advisory Synthesis & ICAR Protocol Generation]
```

---

## 2. Realistic Field Data Augmentations

Training images receive realistic mobile-camera augmentations to prepare the model for varying daylight, smartphone cameras, and field conditions:

- `RandomResizedCrop(224, scale=(0.8, 1.0))` (handheld distance variation)
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(degrees=(-15, 15))` (slight phone tilt)
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)` (sunlight, shadows, overcast days)
- `GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))` (slight camera defocus)
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

Validation and test sets use clean, deterministic resizing and normalization without artificial distortion.

---

## 3. Step-by-Step Training Commands

### Step 1: Ingest & Split Dataset
```powershell
python training/download_dataset.py
python training/prepare_dataset.py
```

### Step 2: Train Crop Classifier
Trains the MobileNetV3 model to distinguish Paddy, Cotton, Tomato, Wheat, Chilli, Potato, etc.:
```powershell
python training/train_crop_model.py --epochs 12 --batch-size 16 --lr 0.001
```
Outputs:
- `models/crop_model.pth`
- `models/crop_model.onnx`
- `models/crop_classes.json`

### Step 3: Train Disease Classifiers
Trains both the unified multi-class disease model and crop-specific specialist models:
```powershell
python training/train_disease_model.py --epochs 12 --batch-size 16 --lr 0.001
```
Outputs:
- `models/disease/unified_disease_model.pth`
- `models/disease/unified_disease_model.onnx`
- `models/disease/<crop>_disease_model.pth`
- `models/disease/disease_classes.json`

---

## 4. Model Evaluation & Reports

Evaluate all trained models against the held-out test split:

```powershell
python training/evaluate_model.py
```

Outputs stored in `reports/`:
- `reports/evaluation_report.json`: Full precision, recall, F1-scores, accuracy, and support.
- `reports/classification_report.txt`: Human-readable text summary of per-class performance.
- `reports/confusion_matrix.json`: Numerical confusion matrices for crop and disease classifiers.

---

## 5. Exporting & Inference Runtime

- **ONNX Export**: All training scripts automatically export models to ONNX (opset 14) for fast CPU/GPU inference via `onnxruntime`.
- **TFLite Conversion**: Models can also be converted to TFLite for edge Android/iOS deployments using TensorFlow or `ai-edge-torch`.
