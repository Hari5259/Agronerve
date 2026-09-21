# AgroNerve Vision Dataset Setup Guide

This document outlines how real agricultural crop disease datasets are acquired, organized, and verified within AgroNerve.

---

## 1. Directory Structure

The dataset structure follows strict train, validation, and test partitioning:

```
data/vision_dataset/
├── raw/                         # Raw staged images normalized by canonical class
│   ├── paddy_blast/
│   ├── paddy_bacterial_leaf_blight/
│   ├── paddy_sheath_blight/
│   ├── paddy_healthy/
│   ├── cotton_bacterial_blight/
│   ├── cotton_leaf_curl_virus/
│   ├── cotton_healthy/
│   ├── tomato_early_blight/
│   ├── tomato_late_blight/
│   ├── tomato_leaf_curl_virus/
│   ├── tomato_healthy/
│   ├── wheat_yellow_rust/
│   ├── wheat_healthy/
│   ├── chilli_anthracnose/
│   ├── chilli_healthy/
│   ├── potato_early_blight/
│   ├── potato_late_blight/
│   └── potato_healthy/
├── train/                       # ~70% of verified images with training augmentations
├── validation/                  # ~15% of verified images for hyperparameter tuning
├── test/                        # ~15% held-out test partition for final evaluation
└── metadata/
    ├── dataset.csv              # Full dataset manifest with image IDs, paths & labels
    ├── class_names.json         # Canonical disease & healthy class label map
    ├── crop_names.json          # Crop class label map
    ├── download_summary.json    # Ingestion logs & source tally
    └── split_summary.json       # Split count summary per class
```

---

## 2. Public Dataset Sources

AgroNerve utilizes verified, publicly licensed agricultural vision corpora:

1. **Digital Green Smallholder Farmer Crop Photographs** (`DigiGreen/Crop_Disease_Images`):
   - Real-world field images captured on mobile phones by smallholder farmers across India.
   - Expert-verified agronomic annotations across Paddy, Wheat, Chilli, Potato, Tomato, Cotton, and Maize.
2. **PlantVillage Agricultural Corpus** (`Viraj77/tomato-disease-dataset`, `PlantVillage`):
   - High-resolution foliar pathology dataset covering Tomato (Early Blight, Late Blight, Leaf Curl, Healthy), Potato (Early Blight, Late Blight, Healthy), and Pepper/Chilli.
3. **ICAR & Agricultural Research Leaf Collections**:
   - Field pathology datasets covering Paddy blast, bacterial leaf blight, sheath blight, and wheat yellow rust.

---

## 3. How to Run Automated Dataset Download

Run the automated ingestion pipeline:

```powershell
python training/download_dataset.py --max-per-class 200
```

This script:
1. Dynamically discovers all crops and diseases configured in `data/knowledge_base/disease.json`.
2. Streams verified images from HuggingFace and public repositories.
3. Validates image integrity using PIL (`Image.verify()` and RGB mode conversion).
4. Discards corrupted or unusable image files.
5. Computes MD5 checksums to deduplicate identical images.
6. Saves normalized JPGs into `data/vision_dataset/raw/<class_name>/`.

---

## 4. How to Add Custom or Offline Datasets

If you have downloaded an offline archive (e.g. Kaggle PlantVillage, PlantDoc, or ICAR field archives):

1. Create directory `data/raw_datasets/` (if not already present).
2. Place your image folders or uncompressed folders inside `data/raw_datasets/`.
   - Example folder naming: `data/raw_datasets/Tomato___Early_blight/`, `data/raw_datasets/rice_blast/`, etc.
3. Run the download and ingestion script:
   ```powershell
   python training/download_dataset.py
   ```
4. The scanner will automatically detect, normalize, validate, and deduplicate your local images into the AgroNerve structure.

---

## 5. Preparing Train / Validation / Test Splits

Once raw images are staged in `data/vision_dataset/raw/`, generate the stratified 70/15/15 splits:

```powershell
python training/prepare_dataset.py --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
```

This generates `data/vision_dataset/metadata/dataset.csv` and class indices in `class_names.json`.
