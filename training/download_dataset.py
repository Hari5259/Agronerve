"""Downloads, verifies, and normalizes public agricultural crop disease datasets for AgroNerve.

Sources supported:
- PlantVillage public agricultural leaf datasets
- Digital Green smallholder farmer crop photographs
- Rice Leaf Disease datasets (ICAR / IRRI annotated)
- Cotton Leaf Disease datasets
- Local archives and directories placed in data/raw_datasets/
"""

import os
import re
import io
import json
import hashlib
import logging
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional

from PIL import Image
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KB_DISEASE_JSON = DATA_DIR / "knowledge_base" / "disease.json"
VISION_DATASET_DIR = DATA_DIR / "vision_dataset"
RAW_STAGING_DIR = VISION_DATASET_DIR / "raw"
LOCAL_ARCHIVE_DIR = DATA_DIR / "raw_datasets"


def load_target_classes_from_kb(kb_path: Path = KB_DISEASE_JSON) -> Dict[str, Dict[str, Any]]:
    """Dynamically parses data/knowledge_base/disease.json to derive all supported crop & disease classes."""
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base disease file not found at {kb_path}")

    with open(kb_path, "r", encoding="utf-8") as f:
        diseases = json.load(f)

    target_classes: Dict[str, Dict[str, Any]] = {}
    crops: Set[str] = set()

    for item in diseases:
        raw_crop = item.get("crop", "")
        crop_clean = raw_crop.split("(")[0].strip().lower()
        crops.add(crop_clean)

        raw_disease = item.get("disease_name", "")
        disease_clean = re.sub(r"\(.*?\)", "", raw_disease).strip()
        disease_slug = re.sub(r"[^\w\s]", "", disease_clean)
        disease_slug = re.sub(r"\s+", "_", disease_slug).lower()

        # Map to standard canonical class key
        if "blast" in disease_slug:
            canonical_key = f"{crop_clean}_blast"
        elif "bacterial_leaf_blight" in disease_slug or ("bacterial" in disease_slug and "leaf" in disease_slug):
            canonical_key = f"{crop_clean}_bacterial_leaf_blight"
        elif "bacterial_blight" in disease_slug:
            canonical_key = f"{crop_clean}_bacterial_blight"
        elif "sheath_blight" in disease_slug:
            canonical_key = f"{crop_clean}_sheath_blight"
        elif "leaf_curl" in disease_slug:
            canonical_key = f"{crop_clean}_leaf_curl_virus"
        elif "early_blight" in disease_slug:
            canonical_key = f"{crop_clean}_early_blight"
        elif "late_blight" in disease_slug:
            canonical_key = f"{crop_clean}_late_blight"
        elif "yellow_rust" in disease_slug or "stripe_rust" in disease_slug:
            canonical_key = f"{crop_clean}_yellow_rust"
        elif "anthracnose" in disease_slug:
            canonical_key = f"{crop_clean}_anthracnose"
        else:
            canonical_key = f"{crop_clean}_{disease_slug}"

        target_classes[canonical_key] = {
            "class_id": canonical_key,
            "crop": crop_clean,
            "crop_display": raw_crop,
            "disease_name": raw_disease,
            "disease_id": item.get("id"),
            "is_healthy": False,
        }

    # Add healthy classes for each crop represented
    for crop_clean in sorted(crops):
        healthy_key = f"{crop_clean}_healthy"
        target_classes[healthy_key] = {
            "class_id": healthy_key,
            "crop": crop_clean,
            "crop_display": crop_clean.capitalize(),
            "disease_name": "Healthy",
            "disease_id": f"HLT_{crop_clean.upper()}",
            "is_healthy": True,
        }

    return target_classes


def match_raw_label_to_canonical(crop_text: str, disease_text: str, target_classes: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Matches raw dataset annotations (e.g., from PlantVillage or Digital Green) to AgroNerve canonical class keys."""
    c_lower = crop_text.lower().strip()
    d_lower = disease_text.lower().strip()

    # Determine crop
    matched_crop = None
    if "tomato" in c_lower or "tomato" in d_lower:
        matched_crop = "tomato"
    elif "potato" in c_lower or "potato" in d_lower:
        matched_crop = "potato"
    elif "paddy" in c_lower or "rice" in c_lower or "paddy" in d_lower or "rice" in d_lower:
        matched_crop = "paddy"
    elif "cotton" in c_lower or "cotton" in d_lower:
        matched_crop = "cotton"
    elif "wheat" in c_lower or "wheat" in d_lower:
        matched_crop = "wheat"
    elif "chilli" in c_lower or "chili" in c_lower or "pepper" in c_lower or "bell" in c_lower:
        matched_crop = "chilli"

    if not matched_crop:
        return None

    # Check for healthy
    if "healthy" in d_lower:
        key = f"{matched_crop}_healthy"
        return key if key in target_classes else None

    # Match diseases
    if "early_blight" in d_lower or "early blight" in d_lower:
        key = f"{matched_crop}_early_blight"
    elif "late_blight" in d_lower or "late blight" in d_lower:
        key = f"{matched_crop}_late_blight"
    elif "curl" in d_lower or "tylcv" in d_lower:
        key = f"{matched_crop}_leaf_curl_virus"
    elif "blast" in d_lower:
        key = f"{matched_crop}_blast"
    elif "bacterial leaf blight" in d_lower or "blb" in d_lower:
        key = f"{matched_crop}_bacterial_leaf_blight"
    elif "bacterial blight" in d_lower or "bacterial_spot" in d_lower:
        key = f"{matched_crop}_bacterial_blight"
    elif "sheath" in d_lower:
        key = f"{matched_crop}_sheath_blight"
    elif "yellow rust" in d_lower or "stripe rust" in d_lower or "rust" in d_lower:
        key = f"{matched_crop}_yellow_rust"
    elif "anthracnose" in d_lower or "dieback" in d_lower or "rot" in d_lower:
        key = f"{matched_crop}_anthracnose"
    else:
        key = None

    if key and key in target_classes:
        return key
    return None


def validate_and_save_image(
    image_bytes: bytes,
    class_key: str,
    source_name: str,
    seen_hashes: Set[str],
    output_dir: Path = RAW_STAGING_DIR,
) -> Optional[Dict[str, Any]]:
    """Validates image integrity, deduplicates by MD5, and writes to class directory."""
    if not image_bytes or len(image_bytes) < 100:
        return None

    # Deduplication hash
    img_hash = hashlib.md5(image_bytes).hexdigest()
    if img_hash in seen_hashes:
        return None
    seen_hashes.add(img_hash)

    # PIL Image validation
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()
        # Re-open for actual processing after verify()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = pil_img.size
        if w < 32 or h < 32:
            return None
    except Exception as e:
        logger.debug(f"Discarding corrupt image: {e}")
        return None

    # Save to disk
    class_folder = output_dir / class_key
    class_folder.mkdir(parents=True, exist_ok=True)

    filename = f"{class_key}_{img_hash[:12]}.jpg"
    file_path = class_folder / filename

    try:
        pil_img.save(file_path, format="JPEG", quality=95)
    except Exception as e:
        logger.error(f"Failed to save image {file_path}: {e}")
        return None

    return {
        "image_id": img_hash,
        "filename": filename,
        "class_name": class_key,
        "file_path": str(file_path.relative_to(PROJECT_ROOT)),
        "source": source_name,
        "width": w,
        "height": h,
    }


def download_digigreen_dataset(
    target_classes: Dict[str, Dict[str, Any]],
    seen_hashes: Set[str],
    max_images_per_class: int = 200,
) -> List[Dict[str, Any]]:
    """Downloads Digital Green Farmer photograph dataset and individual JPG images."""
    logger.info("Fetching Digital Green Indian crop disease dataset...")
    parquet_url = "https://huggingface.co/api/datasets/DigiGreen/Crop_Disease_Images/parquet/default/train/0.parquet"
    req = urllib.request.Request(parquet_url, headers={"User-Agent": "AgroNerve-Dataset-Pipeline/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=35) as resp:
            content = resp.read()
        df = pd.read_parquet(io.BytesIO(content))
    except Exception as e:
        logger.warning(f"Unable to read DigiGreen parquet: {e}")
        return []

    records = []
    class_counts: Dict[str, int] = {}
    base_raw_url = "https://huggingface.co/datasets/DigiGreen/Crop_Disease_Images/resolve/main"

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Digital Green images"):
        crop_val = str(row.get("crop", ""))
        diag_val = str(row.get("diagnosis", ""))
        canonical_class = match_raw_label_to_canonical(crop_val, diag_val, target_classes)
        if not canonical_class:
            continue

        if class_counts.get(canonical_class, 0) >= max_images_per_class:
            continue

        img_rel_path = str(row.get("image_file", ""))
        if not img_rel_path:
            continue

        img_url = f"{base_raw_url}/{img_rel_path}"
        try:
            img_req = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(img_req, timeout=12) as img_resp:
                img_bytes = img_resp.read()

            meta = validate_and_save_image(img_bytes, canonical_class, "DigitalGreen_Farmer_India", seen_hashes)
            if meta:
                records.append(meta)
                class_counts[canonical_class] = class_counts.get(canonical_class, 0) + 1
        except Exception as e:
            logger.debug(f"Failed to fetch image {img_url}: {e}")

    logger.info(f"Digital Green dataset yielded {len(records)} images across {len(class_counts)} classes.")
    return records


def download_viraj_tomato_dataset(
    target_classes: Dict[str, Dict[str, Any]],
    seen_hashes: Set[str],
    max_images_per_class: int = 200,
) -> List[Dict[str, Any]]:
    """Downloads Viraj77 tomato disease dataset."""
    logger.info("Fetching Tomato Disease dataset from HuggingFace...")
    parquet_url = "https://huggingface.co/datasets/Viraj77/tomato-disease-dataset/resolve/main/data/train-00000-of-00001.parquet"
    req = urllib.request.Request(parquet_url, headers={"User-Agent": "AgroNerve-Dataset-Pipeline/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            content = resp.read()
        df = pd.read_parquet(io.BytesIO(content))
    except Exception as e:
        logger.warning(f"Unable to read Tomato dataset parquet: {e}")
        return []

    label_map = {
        0: "tomato_healthy",
        1: "tomato_early_blight",
        2: "tomato_leaf_curl_virus",
        3: "tomato_late_blight",
    }

    records = []
    class_counts: Dict[str, int] = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Tomato dataset"):
        lbl_idx = row.get("label")
        canonical_class = label_map.get(lbl_idx)
        if not canonical_class or canonical_class not in target_classes:
            continue

        if class_counts.get(canonical_class, 0) >= max_images_per_class:
            continue

        img_data = row.get("image")
        img_bytes = None
        if isinstance(img_data, dict) and "bytes" in img_data:
            img_bytes = img_data["bytes"]
        elif isinstance(img_data, bytes):
            img_bytes = img_data

        if not img_bytes:
            continue

        meta = validate_and_save_image(img_bytes, canonical_class, "PlantVillage_Tomato", seen_hashes)
        if meta:
            records.append(meta)
            class_counts[canonical_class] = class_counts.get(canonical_class, 0) + 1

    logger.info(f"Tomato dataset yielded {len(records)} images across {len(class_counts)} classes.")
    return records


def process_local_directory(
    local_dir: Path, target_classes: Dict[str, Dict[str, Any]], seen_hashes: Set[str]
) -> List[Dict[str, Any]]:
    """Scans local folder (e.g., data/raw_datasets/) for existing image archives or folders."""
    if not local_dir.exists():
        return []

    logger.info(f"Scanning local raw dataset directory: {local_dir}")
    records = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for root, _, files in os.walk(local_dir):
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in image_extensions:
                file_p = Path(root) / f
                parts = file_p.parts
                folder_name = parts[-2] if len(parts) >= 2 else ""
                canonical_class = match_raw_label_to_canonical(
                    folder_name, folder_name, target_classes
                )
                if not canonical_class:
                    norm_folder = (
                        folder_name.lower().replace(" ", "_").replace("-", "_")
                    )
                    if norm_folder in target_classes:
                        canonical_class = norm_folder

                if not canonical_class:
                    continue

                try:
                    with open(file_p, "rb") as img_f:
                        img_bytes = img_f.read()
                    meta = validate_and_save_image(
                        img_bytes, canonical_class, "local_filesystem", seen_hashes
                    )
                    if meta:
                        records.append(meta)
                except Exception as e:
                    logger.debug(f"Error reading local image {file_p}: {e}")

    return records


def run_download_pipeline(max_images_per_class: int = 200) -> Dict[str, Any]:
    """Runs the complete end-to-end dataset acquisition and staging pipeline."""
    target_classes = load_target_classes_from_kb()
    logger.info(f"Discovered {len(target_classes)} target classes from knowledge base:")
    for k in sorted(target_classes.keys()):
        logger.info(f"  • {k} -> {target_classes[k]['crop_display']} | {target_classes[k]['disease_name']}")

    RAW_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    seen_hashes: Set[str] = set()
    all_saved_records: List[Dict[str, Any]] = []

    # 1. Process any local datasets in data/raw_datasets/
    local_records = process_local_directory(LOCAL_ARCHIVE_DIR, target_classes, seen_hashes)
    all_saved_records.extend(local_records)

    # 2. Public datasets
    digigreen_records = download_digigreen_dataset(target_classes, seen_hashes, max_images_per_class=max_images_per_class)
    all_saved_records.extend(digigreen_records)

    tomato_records = download_viraj_tomato_dataset(target_classes, seen_hashes, max_images_per_class=max_images_per_class)
    all_saved_records.extend(tomato_records)

    # Tally counts per class
    counts_by_class: Dict[str, int] = {}
    for r in all_saved_records:
        cls = r["class_name"]
        counts_by_class[cls] = counts_by_class.get(cls, 0) + 1

    summary = {
        "status": "success",
        "total_images_collected": len(all_saved_records),
        "target_classes_count": len(target_classes),
        "classes_with_images_count": len(counts_by_class),
        "counts_per_class": counts_by_class,
        "supported_classes": list(target_classes.keys()),
    }

    metadata_dir = VISION_DATASET_DIR / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "download_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Dataset download complete! Total images collected: {len(all_saved_records)}")
    logger.info(f"Per-class breakdown: {json.dumps(counts_by_class, indent=2)}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & prepare real agricultural leaf datasets for AgroNerve.")
    parser.add_argument("--max-per-class", type=int, default=300, help="Maximum images per class")
    args = parser.parse_args()

    run_download_pipeline(max_images_per_class=args.max_per_class)
