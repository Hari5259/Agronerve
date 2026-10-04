"""AgroNerve Agricultural Calculations and Precision Agronomy Engine.

Provides mathematical algorithms for chemical dilution, knapsack/tractor spray volume calculation,
irrigation water requirement estimation (FAO-56 ETc), and fertilizer NPK split planning.
"""

from typing import Dict, Any, List, Optional
import math

# FAO-56 Crop Coefficients (Kc) for key crops across growth stages
CROP_KC_VALUES: Dict[str, Dict[str, float]] = {
    "paddy": {"initial": 1.05, "mid": 1.20, "late": 0.90},
    "rice": {"initial": 1.05, "mid": 1.20, "late": 0.90},
    "tomato": {"initial": 0.60, "mid": 1.15, "late": 0.80},
    "cotton": {"initial": 0.35, "mid": 1.20, "late": 0.65},
    "wheat": {"initial": 0.40, "mid": 1.15, "late": 0.40},
    "chilli": {"initial": 0.60, "mid": 1.05, "late": 0.80},
    "maize": {"initial": 0.30, "mid": 1.20, "late": 0.60},
    "banana": {"initial": 1.00, "mid": 1.20, "late": 1.10},
}


class AgriCalculator:
    """Precision agronomic calculations engine."""

    @staticmethod
    def calculate_spray_dosage(
        area_acres: float,
        dose_per_liter: float,
        unit: str = "ml",
        tank_capacity_liters: float = 16.0,
        water_volume_per_acre: float = 200.0,
    ) -> Dict[str, Any]:
        """Calculates exact tank refills, total water volume, and chemical quantity.

        Args:
            area_acres: Farm area in acres.
            dose_per_liter: Recommended dosage per liter of water (ml/L or g/L).
            unit: 'ml' or 'g'.
            tank_capacity_liters: Sprayer capacity in liters (default 16L standard knapsack).
            water_volume_per_acre: Standard spray water volume per acre (liters).
        """
        total_water_liters = round(area_acres * water_volume_per_acre, 1)
        total_chemical_quantity = round(total_water_liters * dose_per_liter, 2)
        chemical_per_tank = round(tank_capacity_liters * dose_per_liter, 2)
        total_tanks = math.ceil(total_water_liters / tank_capacity_liters)

        return {
            "area_acres": area_acres,
            "dose_per_liter": dose_per_liter,
            "unit": unit,
            "tank_capacity_liters": tank_capacity_liters,
            "total_water_volume_liters": total_water_liters,
            "total_chemical_required": total_chemical_quantity,
            "chemical_per_tank": chemical_per_tank,
            "total_knapsack_tanks": total_tanks,
            "advisory": (
                f"For {area_acres} acre(s), prepare approximately {total_tanks} knapsack tanks ({tank_capacity_liters}L each). "
                f"Add {chemical_per_tank} {unit} of chemical per full tank for a total of {total_chemical_quantity} {unit}."
            ),
        }

    @staticmethod
    def calculate_water_requirement(
        crop: str,
        stage: str = "mid",
        area_acres: float = 1.0,
        eto_mm_day: float = 4.5,
        irrigation_efficiency: float = 0.85,
    ) -> Dict[str, Any]:
        """Calculates daily and weekly crop evapotranspiration (ETc) and irrigation water budget.

        Args:
            crop: Crop name (e.g. paddy, tomato, cotton, wheat).
            stage: 'initial', 'mid', or 'late'.
            area_acres: Land area in acres.
            eto_mm_day: Reference evapotranspiration in mm/day.
            irrigation_efficiency: System efficiency (0.90 for drip, 0.65 for surface flood).
        """
        crop_clean = crop.lower().strip()
        kc_dict = CROP_KC_VALUES.get(crop_clean, {"initial": 0.5, "mid": 1.0, "late": 0.7})
        kc = kc_dict.get(stage.lower(), 1.0)

        # ETc (mm/day) = ETo * Kc
        etc_mm_day = round(eto_mm_day * kc, 2)

        # 1 mm of water over 1 acre = 4046.86 Liters
        liters_per_acre_per_day = etc_mm_day * 4046.86
        gross_water_liters_day = round((liters_per_acre_per_day * area_acres) / irrigation_efficiency, 0)
        weekly_water_liters = round(gross_water_liters_day * 7, 0)

        return {
            "crop": crop.capitalize(),
            "stage": stage,
            "area_acres": area_acres,
            "kc": kc,
            "etc_mm_day": etc_mm_day,
            "daily_gross_liters": gross_water_liters_day,
            "weekly_gross_liters": weekly_water_liters,
            "advisory": (
                f"Crop {crop.capitalize()} at '{stage}' stage consumes approx {etc_mm_day} mm/day. "
                f"Apply ~{gross_water_liters_day:,.0f} Liters of water daily ({weekly_water_liters:,.0f} L/week) for {area_acres} acre(s)."
            ),
        }

    @staticmethod
    def calculate_fertilizer_npk_sources(
        n_kg: float,
        p_kg: float,
        k_kg: float,
    ) -> Dict[str, float]:
        """Calculates commercial fertilizer product weights (Urea, DAP, MOP) to meet N-P-K nutrient targets.

        DAP: 18% N, 46% P2O5
        Urea: 46% N
        MOP: 60% K2O
        """
        # Step 1: P from DAP (46% P2O5)
        dap_kg = (p_kg / 0.46) if p_kg > 0 else 0.0
        n_from_dap = dap_kg * 0.18

        # Step 2: Remaining N from Urea (46% N)
        remaining_n = max(0.0, n_kg - n_from_dap)
        urea_kg = (remaining_n / 0.46) if remaining_n > 0 else 0.0

        # Step 3: K from MOP (60% K2O)
        mop_kg = (k_kg / 0.60) if k_kg > 0 else 0.0

        return {
            "dap_kg": round(dap_kg, 1),
            "urea_kg": round(urea_kg, 1),
            "mop_kg": round(mop_kg, 1),
        }


agri_calculator = AgriCalculator()
