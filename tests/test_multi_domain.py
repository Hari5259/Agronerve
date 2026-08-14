import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from core.multi_domain_router import MultiDomainRouter


def test_single_domain_identification():
    router = MultiDomainRouter()
    res = router.analyze_multi_domain("My paddy has brown spots and blast symptoms")
    assert res["is_multi_domain"] is False
    assert res["primary_domain"] == "disease"


def test_multi_domain_composite_identification():
    router = MultiDomainRouter()
    res = router.analyze_multi_domain(
        "My tomato has wilting disease and heavy rain is forecasted, should I irrigate or spray fungicide?"
    )
    assert res["is_multi_domain"] is True
    assert len(res["active_domains"]) >= 2
    assert "disease" in res["active_domains"]
