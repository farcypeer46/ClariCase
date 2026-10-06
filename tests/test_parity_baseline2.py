"""Parity: our utils reproduce baseline 2's committed numbers exactly.

If this fails, STOP — every improved model comparison would be against a
different denominator than the served baseline.
"""
import json

import pandas as pd
import pytest

from src.utils import config as c
from src.utils.routing import per_team_routing, routed_stats


def _load_preds():
    return pd.read_csv(c.BASELINE2_METRICS / "predictions_test.csv",
                       dtype={c.ID_COL: "string"})


@pytest.mark.slow
def test_routing_reproduces_baseline_2_numbers():
    df = _load_preds()
    ns = json.loads((c.BASELINE2_METRICS / "north_star.json").read_text())
    thr = ns["operating_points"]["at_95p"]["threshold"]
    conf = df["confidence"].to_numpy()
    correct = (df["predicted_team_id"] == df["actual_team_id"]).to_numpy()
    stats = routed_stats(conf, correct, thr)
    assert abs(stats["coverage"] - 0.4646) < 0.001
    assert abs(stats["precision"] - 0.9444) < 0.001


@pytest.mark.slow
def test_per_team_routing_matches_committed_csv():
    """Reuse the committed `auto_routed` boolean rather than recomputing from
    rounded confidence. Baseline 2 wrote `auto_routed` from UNROUNDED
    confidence then rounded the `confidence` column to 4 decimals, so
    `confidence >= 0.872` on the CSV disagrees with `auto_routed` on ~30
    rows (CSV-rounding drift). Using the auto_routed column directly is the
    faithful reproduction of baseline 2's inputs."""
    import numpy as np
    df = _load_preds()
    names = pd.read_csv(c.TEAM_MAPPING)
    names_lookup = dict(zip(names["team_id"], names["team_name"]))
    # Construct synthetic (conf, threshold) that reproduces auto_routed exactly.
    routed = df["auto_routed"].astype(bool).to_numpy()
    synth_conf = np.where(routed, 1.0, 0.0)
    ours = per_team_routing(
        df["actual_team_id"].to_numpy(),
        df["predicted_team_id"].to_numpy(),
        synth_conf, 0.5, names_lookup)
    committed = pd.read_csv(c.BASELINE2_METRICS / "per_team_routing.csv")
    for col in ("team_id", "n_routed_to", "precision_routed",
                "auto_routed_share"):
        pd.testing.assert_series_equal(
            ours.sort_values("team_id").reset_index(drop=True)[col],
            committed.sort_values("team_id").reset_index(drop=True)[col],
            check_dtype=False, atol=1e-4, rtol=0)


@pytest.mark.slow
def test_baseline2_model_scored_on_rebuilt_test_matches_macro_f1():
    import joblib
    from sklearn.metrics import f1_score
    model = joblib.load(c.BASELINE2_METRICS / "model.joblib")
    test = pd.read_parquet(c.SPLITS_V2 / "test.parquet")
    y = test[c.TEAM_COL].to_numpy()
    pred = model.predict(test[c.TEXT_COL].astype(str))
    macro = f1_score(y, pred, average="macro")
    assert abs(macro - 0.6475) < 0.001
