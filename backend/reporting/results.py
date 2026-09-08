"""
Formatting analysis results and hypothesis rankings into structured report structures.
"""
from typing import List, Dict, Any, Optional
try:
    from backend.hypothesis.candidates import HypothesisCandidate
except ImportError:
    from hypothesis.candidates import HypothesisCandidate


def format_hypotheses_report(
    ranked_candidates: List[HypothesisCandidate],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Formats ranked hypotheses into a structured report dictionary.
    """
    report = {
        "metadata": metadata or {},
        "total_hypotheses": len(ranked_candidates),
        "top_hypothesis": ranked_candidates[0].to_dict() if ranked_candidates else None,
        "rankings": [cand.to_dict() for cand in ranked_candidates],
    }
    return report
