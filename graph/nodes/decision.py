"""Decision node — flag, continue, or abstain."""

from __future__ import annotations

from agents.decision import DecisionOutcome
from graph.state import FerretGraphDeps, FerretState, get_deps

try:
    from langgraph.graph import END
except ImportError:  # pragma: no cover
    END = "__end__"


def decision_node(state: FerretState, deps: FerretGraphDeps | None = None, config: dict | None = None) -> FerretState:
    deps = get_deps(config, deps=deps)
    episode = state["episode"]
    if episode is None:
        raise RuntimeError("decision_node requires initialized episode.")

    outcome = deps.decision_node.evaluate(episode.confidence, episode.remaining_budget)
    done = outcome != DecisionOutcome.CONTINUE
    flagged = outcome == DecisionOutcome.FLAG if done else None

    return {
        "decision": outcome.value,
        "done": done,
        "flagged": flagged,
    }


def route_after_decision(state: FerretState):
    if state.get("done"):
        return END
    return "policy"
