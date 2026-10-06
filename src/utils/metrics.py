"""Unified evaluation for team + issue + joint + oracle + hard-pair recall."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)


def bootstrap_ci_macro_f1(y_true: np.ndarray, y_pred: np.ndarray,
                          n_boot: int = 1000, seed: int = 42,
                          ) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    out = np.empty(n_boot)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        out[b] = f1_score(y_true[idx], y_pred[idx],
                          average="macro", zero_division=0)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _top_k_accuracy(y_true, scores, classes, k: int = 3) -> float:
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    # Index of true label in `classes`; -1 means label not seen in classes.
    idx = {c: i for i, c in enumerate(classes)}
    true_idx = np.array([idx.get(t, -1) for t in y_true])
    valid = true_idx >= 0
    if not valid.any():
        return 0.0
    top_k = np.argsort(-scores[valid], axis=1)[:, :k]
    return float((top_k == true_idx[valid, None]).any(axis=1).mean())


def _hard_pair_metrics(y_team_true, y_team_pred) -> dict:
    y_true = np.asarray(y_team_true)
    y_pred = np.asarray(y_team_pred)
    def _recall(tid):
        own = y_true == tid
        return float((y_pred[own] == tid).mean()) if own.any() else None
    def _share_predicted_as(tid, other):
        own = y_true == tid
        if not own.any():
            return None
        return float((y_pred[own] == other).mean())
    return {
        "T02_recall": _recall("T02"),
        "T02_as_T01_share": _share_predicted_as("T02", "T01"),
        "T05_recall": _recall("T05"),
        "T05_as_T04_share": _share_predicted_as("T05", "T04"),
        "T11_recall": _recall("T11"),
        "T11_as_T01_share": _share_predicted_as("T11", "T01"),
        "T11_as_T02_share": _share_predicted_as("T11", "T02"),
    }


def _team_block(y_team, team_pred, team_classes, bootstrap: bool, seed: int):
    labels = sorted(set(team_classes) | set(y_team) | set(team_pred))
    report = classification_report(y_team, team_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    cmat = confusion_matrix(y_team, team_pred, labels=labels).tolist()
    block = {
        "accuracy": float(accuracy_score(y_team, team_pred)),
        "macro_f1": float(f1_score(y_team, team_pred, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_team, team_pred, average="weighted",
                                      zero_division=0)),
        "per_class": {k: v for k, v in report.items()
                      if k not in ("accuracy", "macro avg", "weighted avg")},
        "confusion_matrix": {"labels": labels, "matrix": cmat},
    }
    if bootstrap:
        lo, hi = bootstrap_ci_macro_f1(y_team, team_pred, n_boot=1000, seed=seed)
        block["macro_f1_ci_95"] = [lo, hi]
    return block


def _issue_block(y_issue, issue_scores, issue_classes, bootstrap, seed):
    classes = np.asarray(issue_classes)
    pred_idx = np.argmax(issue_scores, axis=1)
    issue_pred = classes[pred_idx]
    block = {
        "accuracy": float(accuracy_score(y_issue, issue_pred)),
        "macro_f1": float(f1_score(y_issue, issue_pred, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_issue, issue_pred, average="weighted",
                                      zero_division=0)),
        "top3_accuracy": _top_k_accuracy(y_issue, issue_scores, classes, k=3),
    }
    labels = sorted(set(classes) | set(y_issue))
    block["per_class"] = {
        k: v for k, v in classification_report(
            y_issue, issue_pred, labels=labels,
            output_dict=True, zero_division=0).items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    }
    if bootstrap:
        lo, hi = bootstrap_ci_macro_f1(y_issue, issue_pred, n_boot=1000, seed=seed)
        block["macro_f1_ci_95"] = [lo, hi]
    return block, issue_pred


def evaluate(y_team, team_proba, team_classes,
             y_issue=None, issue_scores=None, issue_classes=None,
             oracle_issue_scores=None,
             bootstrap: bool = True, seed: int = 42) -> dict:
    """One-stop evaluation for an improved model.

    Required: y_team + team_proba + team_classes.
    Optional (both or none): y_issue + issue_scores + issue_classes.
    Optional: oracle_issue_scores — issue scores computed under the TRUE team
    (upper bound for the hierarchy).
    """
    team_classes_arr = np.asarray(team_classes)
    team_pred = team_classes_arr[np.argmax(team_proba, axis=1)]
    team = _team_block(y_team, team_pred, team_classes_arr, bootstrap, seed)
    out: dict = {"team": team, "issue": None, "joint": None,
                 "oracle": None, "hard_pairs": _hard_pair_metrics(y_team, team_pred)}

    if y_issue is None or issue_scores is None or issue_classes is None:
        return out

    issue_block, issue_pred = _issue_block(y_issue, issue_scores, issue_classes,
                                           bootstrap, seed)
    out["issue"] = issue_block
    out["joint"] = {
        "joint_exact": float(np.mean(
            (np.asarray(team_pred) == np.asarray(y_team)) &
            (np.asarray(issue_pred) == np.asarray(y_issue))))
    }

    if oracle_issue_scores is not None:
        oracle_block, _ = _issue_block(y_issue, oracle_issue_scores,
                                       issue_classes, bootstrap, seed)
        out["oracle"] = {
            "accuracy": oracle_block["accuracy"],
            "macro_f1": oracle_block["macro_f1"],
            "top3_accuracy": oracle_block["top3_accuracy"],
        }
    return out
