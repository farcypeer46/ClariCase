"""Hierarchical LinearSVC: team classifier + per-team issue classifier."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.svm import LinearSVC


class _ConstantIssue:
    """Fallback predictor for a team that has only one issue in training."""
    def __init__(self, issue_id: str):
        self.issue_id = issue_id
        self.classes_ = np.array([issue_id])

    def decision_function(self, X):
        return np.ones(X.shape[0], dtype=np.float32)

    def predict(self, X):
        return np.array([self.issue_id] * X.shape[0])


class HierarchicalSVM:
    def __init__(self, C_team: float = 1.0, C_issue: float = 1.0,
                 max_iter: int = 5000):
        self.C_team = C_team
        self.C_issue = C_issue
        self.max_iter = max_iter
        self.team_clf: LinearSVC | None = None
        self.team_classes_: np.ndarray | None = None
        self.issue_clfs_: dict[str, object] = {}
        self.calibrated_team_: CalibratedClassifierCV | None = None
        self.calibrated_issue_: dict[str, object] = {}

    # ---- fit ----
    def fit(self, X, y_team, y_issue) -> "HierarchicalSVM":
        y_team = np.asarray(y_team)
        y_issue = np.asarray(y_issue)
        self.team_clf = LinearSVC(
            C=self.C_team, class_weight="balanced",
            dual="auto", max_iter=self.max_iter,
        )
        self.team_clf.fit(X, y_team)
        self.team_classes_ = self.team_clf.classes_

        self.issue_clfs_ = {}
        for team in self.team_classes_:
            mask = y_team == team
            issues = np.unique(y_issue[mask])
            if len(issues) < 2:
                self.issue_clfs_[team] = _ConstantIssue(issues[0])
                continue
            clf = LinearSVC(C=self.C_issue, class_weight="balanced",
                            dual="auto", max_iter=self.max_iter)
            clf.fit(X[mask], y_issue[mask])
            self.issue_clfs_[team] = clf
        return self

    # ---- prediction ----
    def predict_team(self, X) -> np.ndarray:
        return self.team_clf.predict(X)

    def team_proba(self, X) -> np.ndarray:
        if self.calibrated_team_ is None:
            raise RuntimeError("Call fit_team_calibrator() before team_proba().")
        return self.calibrated_team_.predict_proba(X)

    def predict_issue_topk(self, X, teams, k: int = 3
                           ) -> list[list[tuple[str, float]]]:
        out: list[list[tuple[str, float]]] = []
        for i, team in enumerate(teams):
            clf = self.issue_clfs_[team]
            if isinstance(clf, _ConstantIssue):
                out.append([(clf.issue_id, 1.0)])
                continue
            cal = self.calibrated_issue_.get(team)
            row = X[i]
            if cal is not None:
                proba = cal.predict_proba(row)[0]
                classes = cal.classes_
            else:
                scores = clf.decision_function(row)
                scores = np.atleast_2d(scores).reshape(-1)
                proba = np.exp(scores - scores.max())
                proba = proba / proba.sum()
                classes = clf.classes_
            order = proba.argsort()[::-1][:k]
            out.append([(str(classes[j]), float(proba[j])) for j in order])
        return out

    # ---- calibration ----
    def fit_team_calibrator(self, X_val, y_val_team) -> None:
        self.calibrated_team_ = CalibratedClassifierCV(
            FrozenEstimator(self.team_clf), method="isotonic")
        self.calibrated_team_.fit(X_val, y_val_team)

    def fit_issue_calibrators(self, X_val, y_val_team, y_val_issue,
                              min_rows: int = 200) -> None:
        y_val_team = np.asarray(y_val_team)
        y_val_issue = np.asarray(y_val_issue)
        for team, clf in self.issue_clfs_.items():
            if isinstance(clf, _ConstantIssue):
                continue
            mask = y_val_team == team
            n = int(mask.sum())
            n_issues = len(np.unique(y_val_issue[mask]))
            if n < min_rows or n_issues < 2:
                continue
            method = "isotonic" if n >= min_rows * 2 else "sigmoid"
            cal = CalibratedClassifierCV(FrozenEstimator(clf), method=method)
            cal.fit(X_val[mask], y_val_issue[mask])
            self.calibrated_issue_[team] = cal

    def compact(self) -> None:
        """Cast coef_/intercept_ to float32 to shrink the saved artifact."""
        clfs = [self.team_clf] + list(self.issue_clfs_.values())
        for clf in clfs:
            if isinstance(clf, _ConstantIssue):
                continue
            clf.coef_ = clf.coef_.astype(np.float32)
            clf.intercept_ = clf.intercept_.astype(np.float32)
