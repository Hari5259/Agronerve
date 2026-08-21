import random
import time
from typing import Dict, Any


class SensorTelemetryManager:
    """Manages real-time on-device and simulated soil/weather sensor telemetry."""

    def __init__(self):
        self._last_reading_time = time.time()
        # Initial sensor baseline state
        self._soil_moisture_vwc = 34.5  # Volumetric Water Content %
        self._soil_temp_c = 24.2  # Soil temperature °C
        self._ambient_temp_c = 29.8  # Ambient air temperature °C
        self._ambient_humidity_rh = 68.0  # Relative humidity %
        self._solar_radiation_wm2 = 520  # Solar radiation W/m²
        self._rain_detected = False

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns current live sensor readings with agronomic risk triggers."""
        # Add slight natural jitter for live telemetry simulation
        self._soil_moisture_vwc = max(
            10.0,
            min(85.0, round(self._soil_moisture_vwc + random.uniform(-0.4, 0.4), 1)),
        )
        self._soil_temp_c = round(self._soil_temp_c + random.uniform(-0.1, 0.1), 1)
        self._ambient_temp_c = round(
            self._ambient_temp_c + random.uniform(-0.2, 0.2), 1
        )
        self._ambient_humidity_rh = max(
            20.0,
            min(98.0, round(self._ambient_humidity_rh + random.uniform(-0.5, 0.5), 1)),
        )

        # Agronomic Threshold Evaluations
        irrigation_alert = "Normal Soil Hydration"
        irrigation_status_code = "NORMAL"
        if self._soil_moisture_vwc < 25.0:
            irrigation_alert = "CRITICAL WATER STRESS: Soil moisture below 25% field capacity. Irrigation recommended immediately."
            irrigation_status_code = "CRITICAL_DRY"
        elif self._soil_moisture_vwc > 75.0:
            irrigation_alert = "WATERLOGGING RISK: Soil near saturation (>75%). Suspend irrigation and verify drainage."
            irrigation_status_code = "SATURATED"

        pathogen_risk = "Low"
        if self._ambient_humidity_rh > 85.0 and 20.0 <= self._ambient_temp_c <= 28.0:
            pathogen_risk = "HIGH (Fungal spore germination threshold reached: High RH + Moderate Temp)"
        elif self._ambient_temp_c < 4.0:
            pathogen_risk = "FROST WARNING: Extreme cold hazard. Foliage frost injury expected. Overnight sprinkler irrigation recommended."

        spray_window_status = "Optimal"
        if self._ambient_temp_c > 35.0:
            spray_window_status = (
                "Unfavorable (High temperature will cause droplet scorch)"
            )
        elif self._ambient_temp_c < 10.0:
            spray_window_status = (
                "Unfavorable (Low temperature reduces systemic chemical uptake)"
            )
        elif self._rain_detected:
            spray_window_status = "Prohibited (Rain wash-off risk)"
        elif self._ambient_humidity_rh > 90.0:
            spray_window_status = (
                "Sub-optimal (High humidity delays spray drying, increasing run-off risk)"
            )

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "soil": {
                "moisture_vwc_pct": self._soil_moisture_vwc,
                "temperature_celsius": self._soil_temp_c,
                "status": irrigation_status_code,
                "alert": irrigation_alert,
            },
            "environment": {
                "ambient_temp_celsius": self._ambient_temp_c,
                "relative_humidity_pct": self._ambient_humidity_rh,
                "solar_radiation_wm2": self._solar_radiation_wm2,
                "rain_detected": self._rain_detected,
            },
            "agronomic_advisories": {
                "irrigation_action": irrigation_alert,
                "pathogen_infection_risk": pathogen_risk,
                "chemical_spray_window": spray_window_status,
            },
        }

    def set_manual_telemetry(
        self, soil_moisture: float, ambient_temp: float, humidity: float, rain: bool
    ):
        """Allows manual calibration or testing with custom sensor readings with physical clamping."""
        self._soil_moisture_vwc = max(0.0, min(100.0, float(soil_moisture)))
        self._ambient_temp_c = max(-50.0, min(60.0, float(ambient_temp)))
        self._ambient_humidity_rh = max(0.0, min(100.0, float(humidity)))
        self._rain_detected = bool(rain)


sensor_manager = SensorTelemetryManager()
