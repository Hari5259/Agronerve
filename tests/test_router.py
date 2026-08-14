import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from core.intent_router import IntentRouter


def test_disease_intent_routing():
    router = IntentRouter()
    res = router.route("My rice crop has black blast spots on the leaves")
    assert res["domain"] == "disease"
    assert res["is_confident"] is True


def test_pesticide_intent_routing():
    router = IntentRouter()
    res = router.route(
        "What is the recommended dosage of Chlorantraniliprole insecticide per liter?"
    )
    assert res["domain"] == "pesticide"
    assert res["is_confident"] is True


def test_weather_intent_routing():
    router = IntentRouter()
    res = router.route("Is heavy rain forecasted in tomorrow's weather report?")
    assert res["domain"] == "weather"
    assert res["is_confident"] is True


def test_irrigation_intent_routing():
    router = IntentRouter()
    res = router.route(
        "What is the drip irrigation watering interval for tomato crops?"
    )
    assert res["domain"] == "irrigation"
    assert res["is_confident"] is True


def test_general_fallback():
    router = IntentRouter()
    res = router.route("Hello, what is your purpose?")
    assert res["domain"] == "general"
