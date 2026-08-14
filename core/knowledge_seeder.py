import json
import os
from typing import List, Dict, Any
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"


class KnowledgeChunker:
    """Chunks structured domain JSON data into prompt-retrievable knowledge units."""

    @staticmethod
    def load_disease_chunks() -> List[Dict[str, Any]]:
        file_path = DATA_DIR / "disease.json"
        if not file_path.exists():
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        chunks = []
        for item in items:
            text = (
                f"Crop: {item.get('crop')} | Disease: {item.get('disease_name')}\n"
                f"Symptoms: {item.get('symptoms')}\n"
                f"Favorable Conditions: {item.get('favorable_conditions')}\n"
                f"Management Protocol: {item.get('management')}\n"
                f"Source: {item.get('source')}"
            )
            chunks.append(
                {
                    "id": item.get("id"),
                    "domain": "disease",
                    "crop": item.get("crop"),
                    "title": item.get("disease_name"),
                    "text": text,
                    "metadata": {
                        "type": "disease_diagnosis",
                        "source": item.get("source"),
                    },
                }
            )
        return chunks

    @staticmethod
    def load_pesticide_chunks() -> List[Dict[str, Any]]:
        file_path = DATA_DIR / "pesticide.json"
        if not file_path.exists():
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        chunks = []
        for item in items:
            text = (
                f"Chemical / Formulation: {item.get('trade_or_common_name')}\n"
                f"Category: {item.get('category')}\n"
                f"Target Pests & Diseases: {item.get('target_pests')}\n"
                f"Dosage Rate: {item.get('recommended_dosage')}\n"
                f"Pre-Harvest Interval: {item.get('pre_harvest_interval_days')} days\n"
                f"Safety Precautions & PPE: {item.get('safety_precautions')}\n"
                f"Regulatory Status: {item.get('cibrc_status')}"
            )
            chunks.append(
                {
                    "id": item.get("id"),
                    "domain": "pesticide",
                    "title": item.get("trade_or_common_name"),
                    "text": text,
                    "metadata": {
                        "category": item.get("category"),
                        "phi_days": item.get("pre_harvest_interval_days"),
                    },
                }
            )
        return chunks

    @staticmethod
    def load_weather_chunks() -> List[Dict[str, Any]]:
        file_path = DATA_DIR / "weather.json"
        if not file_path.exists():
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        chunks = []
        for item in items:
            text = (
                f"Weather Event: {item.get('weather_condition')}\n"
                f"Forecast Window: {item.get('cached_forecast_window')}\n"
                f"Spray Operational Advisory: {item.get('spray_advisory')}\n"
                f"Field & Soil Operations: {item.get('field_operations')}\n"
                f"Disease Risk Alert: {item.get('disease_alert')}"
            )
            chunks.append(
                {
                    "id": item.get("id"),
                    "domain": "weather",
                    "title": item.get("weather_condition"),
                    "text": text,
                    "metadata": {"event": item.get("weather_condition")},
                }
            )
        return chunks

    @staticmethod
    def load_irrigation_chunks() -> List[Dict[str, Any]]:
        file_path = DATA_DIR / "irrigation.json"
        if not file_path.exists():
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        chunks = []
        for item in items:
            stages_text = "\n".join(
                [
                    f"  - {k.replace('_', ' ').title()}: {v}"
                    for k, v in item.get("growth_stages", {}).items()
                ]
            )
            text = (
                f"Crop: {item.get('crop')}\n"
                f"Water Requirement: {item.get('total_water_requirement_mm')}\n"
                f"Growth Stage Watering Schedule:\n{stages_text}\n"
                f"Soil Texture & Dynamics: {item.get('soil_considerations')}\n"
                f"Standard: {item.get('guidelines')}"
            )
            chunks.append(
                {
                    "id": item.get("id"),
                    "domain": "irrigation",
                    "crop": item.get("crop"),
                    "title": f"{item.get('crop')} Irrigation Guidelines",
                    "text": text,
                    "metadata": {"water_req": item.get("total_water_requirement_mm")},
                }
            )
        return chunks

    @classmethod
    def get_all_chunks(cls) -> List[Dict[str, Any]]:
        return (
            cls.load_disease_chunks()
            + cls.load_pesticide_chunks()
            + cls.load_weather_chunks()
            + cls.load_irrigation_chunks()
        )


if __name__ == "__main__":
    chunks = KnowledgeChunker.get_all_chunks()
    print(f"Successfully processed {len(chunks)} knowledge chunks across 4 domains.")
