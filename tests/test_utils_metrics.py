import numpy as np

from src.utils.metrics import bootstrap_ci_macro_f1, evaluate


def _fake():
    y_true = np.array(["T01", "T01", "T02", "T02", "T03"])
    proba = np.array([
        [.7, .2, .1],
        [.3, .4, .3],
        [.1, .8, .1],
        [.2, .6, .2],
        [.1, .2, .7],
    ])
    classes = np.array(["T01", "T02", "T03"])
    return y_true, proba, classes


def test_evaluate_team_only_returns_expected_keys():
    y, p, classes = _fake()
    out = evaluate(y_team=y, team_proba=p, team_classes=classes)
    assert set(out["team"]).issuperset(
        {"accuracy", "macro_f1", "weighted_f1", "per_class",
         "confusion_matrix", "macro_f1_ci_95"})
    assert out.get("issue") is None
    assert out["team"]["accuracy"] > 0.5


def test_bootstrap_ci_returns_lo_hi():
    y, p, classes = _fake()
    pred = classes[p.argmax(1)]
    lo, hi = bootstrap_ci_macro_f1(y, pred, n_boot=50, seed=0)
    assert 0 <= lo <= hi <= 1


def test_evaluate_with_issue_top3_accuracy():
    y_team = np.array(["T01", "T01", "T02"])
    team_proba = np.array([[.6, .4], [.5, .5], [.3, .7]])
    team_classes = np.array(["T01", "T02"])
    y_issue = np.array(["T01_I001", "T01_I001", "T02_I001"])
    issue_scores = np.array([
        [.5, .3, .2],
        [.4, .4, .2],
        [.1, .2, .7],
    ])
    issue_classes = np.array(["T01_I001", "T01_I002", "T02_I001"])
    out = evaluate(y_team=y_team, team_proba=team_proba,
                   team_classes=team_classes,
                   y_issue=y_issue, issue_scores=issue_scores,
                   issue_classes=issue_classes)
    assert "issue" in out
    assert "top3_accuracy" in out["issue"]
    assert "joint_exact" in out["joint"]
