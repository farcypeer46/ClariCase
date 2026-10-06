"""Synthetic-data tests for HierarchicalSVM."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.improved_model_1.model import HierarchicalSVM


@pytest.fixture
def toy():
    rows = []
    for tid in ("T01", "T02", "T03"):
        for iid in (f"{tid}_I001", f"{tid}_I002"):
            for _ in range(15):
                rows.append((tid, iid))
    df = pd.DataFrame(rows, columns=["team_id", "issue_id"])
    rng = np.random.default_rng(0)
    X = np.zeros((len(df), 6), dtype=np.float32)
    for i, t in enumerate(df.team_id):
        col = {"T01": 0, "T02": 1, "T03": 2}[t]
        X[i, col] = 1.0
        if df.issue_id.iloc[i].endswith("I002"):
            X[i, 3 + col] = 1.0
        X[i] += rng.normal(scale=0.05, size=6).astype(np.float32)
    return sparse.csr_matrix(X), df.team_id.to_numpy(), df.issue_id.to_numpy()


def test_fit_predict_team(toy):
    X, yt, yi = toy
    m = HierarchicalSVM(C_team=1.0, C_issue=1.0).fit(X, yt, yi)
    pred = m.predict_team(X)
    assert (pred == yt).mean() > 0.9


def test_predict_issue_topk_shape(toy):
    X, yt, yi = toy
    m = HierarchicalSVM(C_team=1.0, C_issue=1.0).fit(X, yt, yi)
    top = m.predict_issue_topk(X, teams=yt, k=2)
    assert len(top) == X.shape[0]
    assert all(len(r) <= 2 for r in top)


def test_single_issue_team_gets_constant_predictor(toy):
    X, yt, yi = toy
    yi = np.where(yt == "T03", "T03_I001", yi)
    m = HierarchicalSVM(C_team=1.0, C_issue=1.0).fit(X, yt, yi)
    idx = np.where(yt == "T03")[0]
    top = m.predict_issue_topk(X[idx], teams=yt[idx], k=1)
    assert all(r[0][0] == "T03_I001" for r in top)


def test_calibrator_enables_team_proba(toy):
    X, yt, yi = toy
    m = HierarchicalSVM(C_team=1.0, C_issue=1.0).fit(X, yt, yi)
    m.fit_team_calibrator(X, yt)
    proba = m.team_proba(X)
    assert proba.shape == (X.shape[0], 3)
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-3)
