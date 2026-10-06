import numpy as np

from src.utils.routing import (
    apply_per_team_thresholds, coverage_curve, per_team_thresholds,
    routed_stats, threshold_for,
)


def _toy():
    # 20 rows, 2 teams. Confidence sorted desc; first 15 correct.
    conf = np.linspace(0.95, 0.05, 20)
    pred = np.array(["T01"] * 10 + ["T02"] * 10)
    true = np.array(["T01"] * 10 + ["T02"] * 5 + ["T01"] * 5)
    correct = pred == true
    return conf, pred, true, correct


def test_coverage_curve_monotonic_coverage():
    conf, _, _, correct = _toy()
    curve = coverage_curve(conf, correct, min_routed=1)
    assert (curve["coverage"].diff().dropna() <= 0).all()


def test_threshold_for_50pct_reachable():
    conf, _, _, correct = _toy()
    curve = coverage_curve(conf, correct, min_routed=1)
    assert threshold_for(curve, 0.5) is not None


def test_per_team_thresholds_handles_unreachable():
    conf, pred, true, _ = _toy()
    # Target 0.99 so no team can reach it with 20 rows → None.
    thresholds = per_team_thresholds(conf, pred, true, target=0.99, min_routed=1)
    assert set(thresholds) == {"T01", "T02"}


def test_apply_per_team_thresholds_returns_structure():
    conf, pred, true, _ = _toy()
    thresholds = per_team_thresholds(conf, pred, true, target=0.5, min_routed=1)
    out = apply_per_team_thresholds(true, pred, conf, thresholds,
                                    names={"T01": "A", "T02": "B"},
                                    min_routed=1)
    for key in ("coverage", "precision", "per_team"):
        assert key in out


def test_routed_stats_basic():
    conf, _, _, correct = _toy()
    r = routed_stats(conf, correct, threshold=0.5)
    assert 0.0 <= r["coverage"] <= 1.0
