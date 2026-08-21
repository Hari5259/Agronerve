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


def test_empty_query_routing():
    router = IntentRouter()
    res1 = router.route("")
    res2 = router.route("   ")
    assert res1["domain"] == "general"
    assert res1["is_confident"] is True
    assert res2["domain"] == "general"
    assert res2["is_confident"] is True


def test_truncated_long_query_routing():
    router = IntentRouter()
    long_query = "weather " * 200  # 1600 characters
    res = router.route(long_query)
    assert res["domain"] == "weather"
    assert res["is_confident"] is True


def test_expanded_keyword_routing():
    router = IntentRouter()
    # Scab should route to disease
    res_disease = router.route("My apple crop is showing severe scab lesions")
    assert res_disease["domain"] == "disease"
    
    # Vermicompost should route to pesticide (organic soil/pest control/fertilizer context)
    res_pest = router.route("Can I use vermicompost and organic repellents?")
    assert res_pest["domain"] == "pesticide"

    # Cyclone should route to weather
    res_weather = router.route("Is there a cyclone alert for our coastal region?")
    assert res_weather["domain"] == "weather"

    # Drought should route to irrigation
    res_irrig = router.route("How to manage tomato watering during a drought?")
    assert res_irrig["domain"] == "irrigation"

