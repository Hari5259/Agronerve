import re
from typing import List, Dict, Any, Tuple
from core.intent_router import DOMAIN_WEIGHTS, DomainType


class MultiDomainRouter:
    """Detects multi-domain composite queries and orchestrates cross-partition retrieval."""

    def __init__(self, multi_domain_threshold: float = 2.5):
        self.multi_domain_threshold = multi_domain_threshold

    def analyze_multi_domain(self, query: str) -> Dict[str, Any]:
        """Calculates score across all 4 domains to detect compound queries.
        Example: 'My tomato is wilting, should I irrigate or wait for the rain?'
        """
        q_lower = query.lower()
        domain_scores: Dict[str, float] = {d: 0.0 for d in DOMAIN_WEIGHTS}

        for domain, weights in DOMAIN_WEIGHTS.items():
            for word, weight in weights.items():
                pattern = r"\b" + re.escape(word) + r"\b"
                matches = len(re.findall(pattern, q_lower))
                if matches > 0:
                    domain_scores[domain] += weight * matches

        # Sort domains by score
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        primary_domain, primary_score = sorted_domains[0]

        if primary_score == 0.0:
            return {
                "is_multi_domain": False,
                "primary_domain": "general",
                "active_domains": ["general"],
                "domain_scores": domain_scores,
            }

        # Active domains are those exceeding the threshold or reaching >= 40% of primary score
        active_domains = [primary_domain]
        for domain, score in sorted_domains[1:]:
            if score >= self.multi_domain_threshold and score >= (0.35 * primary_score):
                active_domains.append(domain)

        is_multi = len(active_domains) > 1

        return {
            "is_multi_domain": is_multi,
            "primary_domain": primary_domain,
            "active_domains": active_domains,
            "domain_scores": domain_scores,
        }
