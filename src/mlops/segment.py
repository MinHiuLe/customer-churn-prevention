"""
Shared customer segment classification logic.

Both the batch scoring DAG and the real-time API endpoint must use this
module to ensure a customer is never assigned a different segment depending
on whether they were scored offline or online.

Segment definitions
-------------------
Persuadable   : High churn risk AND a retention offer is likely to help.
                Target with vouchers or outreach.
Sleeping_Dog  : A retention offer would *increase* churn probability.
                Suppress — do not contact.
Lost_Cause    : Very high churn risk but the offer has no material effect.
                Escalate to CS; discount spend is wasted here.
Sure_Thing    : Low churn risk regardless of intervention.
                No action required.
"""

from __future__ import annotations


# Thresholds được đặt ở đây để dễ tìm và chỉnh sửa một chỗ duy nhất.
PERSUADABLE_MIN_UPLIFT  = 0.25   # raw uplift score phải dương và đủ lớn
PERSUADABLE_MIN_CHURN   = 0.50   # churn probability tối thiểu để coi là at-risk
SLEEPING_DOG_MAX_UPLIFT = -0.10  # uplift âm: offer làm hại nhiều hơn giúp
LOST_CAUSE_MIN_CHURN    = 0.60   # churn rất cao nhưng offer không hiệu quả

SEGMENT_ACTIONS: dict[str, dict[str, str]] = {
    "Persuadable":  {"action": "Send voucher 20%",      "priority": "HIGH"},
    "Sleeping_Dog": {"action": "Do NOT contact",         "priority": "SUPPRESS"},
    "Lost_Cause":   {"action": "Escalate to CS team",   "priority": "LOW"},
    "Sure_Thing":   {"action": "No action needed",       "priority": "NONE"},
}


def classify_segment(
    uplift_score: float,
    churn_probability: float,
) -> str:
    """
    Classify a single customer into one of four retention segments.

    Parameters
    ----------
    uplift_score      : p(churn | no treatment) − p(churn | treatment).
                        Positive = offer reduces churn.
                        Negative = offer increases churn.
    churn_probability : Model output from LightGBM in [0, 1].

    Returns
    -------
    Segment name as a string.
    """
    if uplift_score >= PERSUADABLE_MIN_UPLIFT and churn_probability >= PERSUADABLE_MIN_CHURN:
        return "Persuadable"
    if uplift_score <= SLEEPING_DOG_MAX_UPLIFT:
        return "Sleeping_Dog"
    if churn_probability >= LOST_CAUSE_MIN_CHURN:
        return "Lost_Cause"
    return "Sure_Thing"


def get_action(segment: str) -> dict[str, str]:
    """Return the recommended action and priority for a given segment."""
    return SEGMENT_ACTIONS.get(segment, {"action": "Unknown", "priority": "NONE"})