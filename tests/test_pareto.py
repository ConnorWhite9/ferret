"""Tests for eval/pareto.py."""

from eval.pareto import ParetoPoint, pareto_frontier


def test_pareto_frontier_basic():
    pts = [
        ParetoPoint("a", accuracy=0.9, mean_probes=3.0),
        ParetoPoint("b", accuracy=0.8, mean_probes=1.0),  # fewer probes, lower acc — on frontier
        ParetoPoint("c", accuracy=0.7, mean_probes=5.0),  # dominated by a
        ParetoPoint("d", accuracy=0.9, mean_probes=4.0),  # dominated by a (same acc, more probes)
    ]
    frontier = pareto_frontier(pts)
    names = {p.detector for p in frontier}
    assert "a" in names  # 0.9 acc, 3 probes — not dominated
    assert "b" in names  # 0.8 acc, 1 probe — fewer probes than a
    assert "c" not in names  # dominated by a (worse acc AND more probes)
    assert "d" not in names  # dominated by a


def test_pareto_single_point():
    pts = [ParetoPoint("only", accuracy=0.7, mean_probes=5.0)]
    frontier = pareto_frontier(pts)
    assert len(frontier) == 1


def test_pareto_all_on_frontier():
    # Monotone trade-off: more probes → higher accuracy, no dominance
    pts = [
        ParetoPoint("fast", accuracy=0.6, mean_probes=1.0),
        ParetoPoint("mid",  accuracy=0.75, mean_probes=5.0),
        ParetoPoint("slow", accuracy=0.9,  mean_probes=10.0),
    ]
    frontier = pareto_frontier(pts)
    assert len(frontier) == 3
