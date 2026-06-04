"""Decision node — flag, continue, or abstain."""

from __future__ import annotations

from enum import Enum

from ferret.constants import DECISION_THRESHOLD


class DecisionOutcome(str, Enum):
    CONTINUE = "continue"
    FLAG = "flag"
    ABSTAIN = "abstain"


class DecisionNode:
    """Threshold confidence and budget to terminate or continue probing."""

    def __init__(self, threshold: float = DECISION_THRESHOLD):
        self.threshold = threshold

    def evaluate(self, confidence: float, remaining_budget: int) -> DecisionOutcome:
        if confidence >= self.threshold:
            return DecisionOutcome.FLAG
        if remaining_budget <= 0:
            return DecisionOutcome.ABSTAIN
        return DecisionOutcome.CONTINUE
