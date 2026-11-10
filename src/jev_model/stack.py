"""Combine Jev's team answer with its yes/no checks into one routing score.

Jev's top probability ties at 1.00 for over half of all complaints, so no
threshold on it reaches 95% precision. The "rich" variant also asks narrow
yes/no questions in the same request (answered independently, so they do not
change the team answer). This script learns, on the validation sample only,
how far those answers agree or conflict with the chosen team, and turns them
into a score that can split the tied group.

Protocol: a logistic regression predicting "Jev's team is correct" is
cross-fitted on validation (5 folds) so the 95%-precision threshold is chosen
on out-of-fold scores; the model is then refit on all of validation and the
frozen model and threshold are applied to test. No API calls are made.

    JEV_MODEL=jev-1.14.0 python -m src.jev_model.stack
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.jev_model.combine import CacheOnly
from src.jev_model.config import (ARTIFACT_DIR, CACHE_DIR, MODEL, N_TEST_DEFAULT,
                                  N_VAL_DEFAULT, NOUL_QUESTIONS, SEED,
                                  TEAM_IDS, cache_path, question_sha)
from src.jev_model.evaluate import classify_frame, load_cache, sample_split
from src.utils.routing import coverage_curve, routed_stats, threshold_for

VARIANT = "rich"
TARGET = 0.95
# The yes/no check that should agree with each team, where one exists.
TEAM_CHECK = {"T01": "credit_report", "T02": "collector",
              "T04": "bank_account", "T05": "payment_app"}


def features(proba: np.ndarray, nouls: list[dict]) -> np.ndarray:
    """Per complaint: Jev's probabilities, the yes/no answers, and whether the
    answers support or contradict the chosen team."""
    s = np.sort(proba, axis=1)
    pred = np.array(TEAM_IDS)[proba.argmax(1)]
    names = list(NOUL_QUESTIONS)
    noul = np.array([[n.get(k, 0.5) for k in names] for n in nouls])
    support = np.array([nl.get(TEAM_CHECK[t], 0.5) if t in TEAM_CHECK else 0.5
                        for t, nl in zip(pred, nouls)])
    conflict = np.array([max([nl.get(c, 0.0) for tm, c in TEAM_CHECK.items()
                              if tm != t] or [0.0])
                         for t, nl in zip(pred, nouls)])
    team_onehot = (pred[:, None] == np.array(TEAM_IDS)[None, :]).astype(float)
    return np.column_stack([s[:, -1], s[:, -1] - s[:, -2], noul,
                            support, conflict, team_onehot])


def _load(df, cache):
    proba = classify_frame(CacheOnly(), df, cache_path(MODEL, VARIANT),
                           qsha=question_sha(VARIANT), legacy_ok=False)
    rows = load_cache(cache_path(MODEL, VARIANT))
    nouls = [rows[str(c)].get("nouls", {}) for c in df["complaint_id"]]
    ok = np.array(TEAM_IDS)[proba.argmax(1)] == df["team_id"].to_numpy()
    return features(proba, nouls), ok, proba.max(1)


def main() -> None:
    from src.utils.data import load_split
    val = sample_split(load_split("val"), N_VAL_DEFAULT)
    test = sample_split(load_split("test"), N_TEST_DEFAULT, seed=SEED + 1)
    Xv, yv, pv = _load(val, None)
    Xt, yt, pt = _load(test, None)

    def model():
        return LogisticRegression(max_iter=2000, C=1.0)

    oof = np.zeros(len(yv))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xv, yv):
        oof[te] = model().fit(Xv[tr], yv[tr]).predict_proba(Xv[te])[:, 1]
    threshold = threshold_for(coverage_curve(oof, yv), TARGET)
    final = model().fit(Xv, yv)
    score_t = final.predict_proba(Xt)[:, 1]

    out = {"variant": VARIANT, "target_precision": TARGET,
           "n_val": len(val), "n_test": len(test),
           "jev_top_probability_only": {
               "best_val_precision": round(float(max(
                   yv[pv >= t].mean() for t in np.unique(pv)
                   if (pv >= t).sum() >= 50)), 4),
               "test_at_1.00": routed_stats(pt, yt, 1.0)},
           "jev_with_yes_no_checks": {
               "threshold_fit_on_validation": threshold,
               "test": None if threshold is None else routed_stats(score_t, yt, threshold)}}
    # Mock output stays in the gitignored cache so it cannot pass for results.
    out_dir = (CACHE_DIR / "mock_artifacts" if MODEL == "mock" else ARTIFACT_DIR) / VARIANT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "yes_no_stack.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
