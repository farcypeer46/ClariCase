# 02 — Improved Model 1: Hierarchical TF-IDF + LinearSVC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a **hierarchical** classifier that predicts the team, then
predicts the issue *within* that team, with isotonic calibration on validation
and both global and per-team thresholds — and beat baseline 2 on the identical
109,834-row test set.

**Architecture:** `HierarchicalSVM` = one `LinearSVC` for team + one
`LinearSVC` per team for issue (fallback constant predictor for single-issue
teams). Features come from `TfidfVectorizer` (word 1–2 + char 3–5) built via
`src.utils.text.clean_text`. Calibration wraps the fitted team classifier with
`CalibratedClassifierCV(FrozenEstimator(...), method="isotonic")` fit on
validation. Two training variants (`cap_issue`, `3k_team`), two validation
protocols (`shared`, `split`), and a flat ablation for the issue head.

**Tech Stack:** scikit-learn 1.9.1 (LinearSVC, CalibratedClassifierCV,
FrozenEstimator, TfidfVectorizer, FeatureUnion), joblib (compress=3), numpy
(float32 coefs).

**Prerequisites:**
- Plan `00-environment-and-repo-hygiene.md` complete (branch, `.gitignore`).
- Plan `01-shared-foundation-utils.md` complete (splits built, parity test
  passing, registry seeded).
- On branch `improved_model_1`.

---

## File layout produced

```
src/improved_model_1/
├── __init__.py
├── config.py       # hyperparameter grid + ARTIFACT_DIR = docs/metrics/improved_model_1
├── features.py     # build_vectorizer(use_char=True, ...)
├── model.py        # HierarchicalSVM class
├── train.py        # entry point
└── predict.py      # CLI + importable predict()
docs/metrics/improved_model_1/
├── model.joblib                      # NOT committed until §5 approves the swap
├── metrics.json
├── classification_report.json
├── confusion_matrix.csv
├── coverage_curve.csv
├── north_star.json
├── per_team_routing.csv
├── predictions_test.csv
├── classification_report_issue.json
├── issue_metrics.json
├── tuning.json
└── run_info.json
docs/improved_model_1.md
```

`reports/model_comparison.csv` gains rows for `improved_model_1`,
`improved_model_1_3k`, `flat_issue_ablation`, `nb_reference_cap_issue`,
`dummy_most_frequent`.

---

### Task 1: Package skeleton

- [ ] **Step 1: Create files**

Write `src/improved_model_1/__init__.py`:
```python
"""Hierarchical TF-IDF + LinearSVC classifier (team → issue)."""
```

- [ ] **Step 2: Commit**

```bash
cd F:/NLP_MSML641 && git add src/improved_model_1/__init__.py
git commit -m "feat(improved_model_1): package skeleton"
```

---

### Task 2: `src/improved_model_1/config.py`

- [ ] **Step 1: Write**

```python
"""Hyperparameter grid and artifact directory for improved_model_1."""
from src.utils.config import METRICS_ROOT

ARTIFACT_DIR = METRICS_ROOT / "improved_model_1"

# Feature settings
USE_CHAR_DEFAULT = True
WORD_MAX_FEATURES = 150_000
CHAR_MAX_FEATURES = 150_000

# Tuning grids (val-metric-driven; §2.5 in the source spec)
C_TEAM_GRID = (0.1, 0.25, 0.5, 1.0)
C_ISSUE_GRID = (0.1, 0.25, 0.5, 1.0)

# Calibration
MIN_ISSUE_ROWS_ISOTONIC = 200
MIN_ISSUES_FOR_MULTICLASS = 2

# Artifact size budget (95 MB)
MAX_ARTIFACT_MB = 95
```

- [ ] **Step 2: Commit**

```bash
cd F:/NLP_MSML641 && git add src/improved_model_1/config.py
git commit -m "feat(improved_model_1): config knobs and artifact dir"
```

---

### Task 3: `src/improved_model_1/features.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_improved_model_1_features.py`:
```python
import numpy as np
from src.improved_model_1.features import build_vectorizer


def test_word_only_vectorizer_fits_and_transforms():
    vec = build_vectorizer(use_char=False, word_max=1000)
    X = vec.fit_transform(["I disputed a debt", "call to collect debt"])
    assert X.shape[0] == 2
    assert X.dtype == np.float32


def test_hybrid_word_char_returns_feature_union():
    vec = build_vectorizer(use_char=True, word_max=200, char_max=200)
    X = vec.fit_transform(["I disputed a debt", "call to collect debt"])
    assert X.shape[0] == 2
```

- [ ] **Step 2: Run test — expect fail**

- [ ] **Step 3: Implement `features.py`**

```python
"""TF-IDF vectorizers using the canonical clean_text as preprocessor."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.utils.text import clean_text


def build_vectorizer(use_char: bool = True,
                     word_max: int = 150_000,
                     char_max: int = 150_000) -> object:
    word = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2), min_df=3, max_df=0.95,
        max_features=word_max, sublinear_tf=True,
        strip_accents="unicode", dtype=np.float32,
    )
    if not use_char:
        return word
    char = TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="char_wb", ngram_range=(3, 5), min_df=5,
        max_features=char_max, sublinear_tf=True, dtype=np.float32,
    )
    return FeatureUnion([("word", word), ("char", char)])
```

- [ ] **Step 4: Verify**

Run: `pytest tests/test_improved_model_1_features.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd F:/NLP_MSML641 && git add src/improved_model_1/features.py \
    tests/test_improved_model_1_features.py
git commit -m "feat(improved_model_1): TF-IDF word+char vectorizer builder"
```

---

### Task 4: `src/improved_model_1/model.py` — `HierarchicalSVM`

- [ ] **Step 1: Write the failing test**

Create `tests/test_improved_model_1_model.py`:
```python
"""Synthetic-data tests for HierarchicalSVM."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.improved_model_1.model import HierarchicalSVM


@pytest.fixture
def toy():
    # 3 teams, each with 2 issues; 90 rows.
    rows = []
    for tid in ("T01", "T02", "T03"):
        for iid in (f"{tid}_I001", f"{tid}_I002"):
            for _ in range(15):
                rows.append((tid, iid))
    df = pd.DataFrame(rows, columns=["team_id", "issue_id"])
    # Fake feature matrix: 2 informative dims per team + noise
    rng = np.random.default_rng(0)
    X = np.zeros((len(df), 6), dtype=np.float32)
    for i, (t, _) in enumerate(zip(df.team_id, df.issue_id)):
        col = {"T01": 0, "T02": 1, "T03": 2}[t]
        X[i, col] = 1.0
        X[i, 3 + col] = 1.0 if df.issue_id.iloc[i].endswith("I002") else 0.0
        X[i] += rng.normal(scale=0.05, size=6).astype(np.float32)
    return sparse.csr_matrix(X), df.team_id.to_numpy(), df.issue_id.to_numpy()


def test_fit_predict_team(toy):
    X, yt, yi = toy
    m = HierarchicalSVM(C_team=1.0, C_issue=1.0)
    m.fit(X, yt, yi)
    pred = m.predict_team(X)
    assert (pred == yt).mean() > 0.9


def test_predict_issue_topk_shape(toy):
    X, yt, yi = toy
    m = HierarchicalSVM(C_team=1.0, C_issue=1.0)
    m.fit(X, yt, yi)
    top = m.predict_issue_topk(X, teams=yt, k=2)
    # top is list[list[tuple[str, float]]] of length n rows
    assert len(top) == X.shape[0]
    assert all(len(r) <= 2 for r in top)


def test_single_issue_team_gets_constant_predictor(toy):
    X, yt, yi = toy
    # Collapse T03 to one issue only
    yi = np.where(yt == "T03", "T03_I001", yi)
    m = HierarchicalSVM(C_team=1.0, C_issue=1.0)
    m.fit(X, yt, yi)
    idx = np.where(yt == "T03")[0]
    top = m.predict_issue_topk(X[idx], teams=yt[idx], k=1)
    assert all(r[0][0] == "T03_I001" for r in top)
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement `src/improved_model_1/model.py`**

```python
"""Hierarchical LinearSVC: team classifier + per-team issue classifier."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.svm import LinearSVC


class _ConstantIssue:
    """Fallback for teams with a single issue in training."""
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

    # --- prediction ---
    def predict_team(self, X) -> np.ndarray:
        return self.team_clf.predict(X)

    def team_proba(self, X) -> np.ndarray:
        if self.calibrated_team_ is None:
            raise RuntimeError("Call fit_team_calibrator() before team_proba().")
        return self.calibrated_team_.predict_proba(X)

    def predict_issue_topk(self, X, teams, k: int = 3
                           ) -> list[list[tuple[str, float]]]:
        out = []
        for i, team in enumerate(teams):
            clf = self.issue_clfs_[team]
            if isinstance(clf, _ConstantIssue):
                out.append([(clf.issue_id, 1.0)])
                continue
            cal = self.calibrated_issue_.get(team)
            if cal is not None:
                proba = cal.predict_proba(X[i])[0]
                classes = cal.classes_
            else:
                scores = clf.decision_function(X[i])
                # softmax over margins as last-resort ordering
                scores = scores.reshape(-1)
                proba = np.exp(scores - scores.max())
                proba = proba / proba.sum()
                classes = clf.classes_
            order = proba.argsort()[::-1][:k]
            out.append([(str(classes[j]), float(proba[j])) for j in order])
        return out

    # --- calibration ---
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
                # Fall back to softmax-over-decision at inference (no calibrator).
                continue
            method = "isotonic" if n >= min_rows * 2 else "sigmoid"
            cal = CalibratedClassifierCV(FrozenEstimator(clf), method=method)
            cal.fit(X_val[mask], y_val_issue[mask])
            self.calibrated_issue_[team] = cal

    # --- artifact size hygiene ---
    def compact(self) -> None:
        """Cast SVC coef_/intercept_ to float32 so joblib(compress=3) fits <95 MB."""
        for clf in [self.team_clf] + list(self.issue_clfs_.values()):
            if isinstance(clf, _ConstantIssue):
                continue
            clf.coef_ = clf.coef_.astype(np.float32)
            clf.intercept_ = clf.intercept_.astype(np.float32)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_improved_model_1_model.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd F:/NLP_MSML641 && git add src/improved_model_1/model.py \
    tests/test_improved_model_1_model.py
git commit -m "feat(improved_model_1): HierarchicalSVM with per-team issue heads"
```

---

### Task 5: `src/improved_model_1/train.py`

**This is the plan's largest task.** Do it in small commits.

- [ ] **Step 1: Scaffold `train.py` with argparse and the load-splits section**

```python
"""Train + evaluate improved_model_1.

Usage:
    python -m src.improved_model_1.train \
        [--train-set cap_issue|3k_team] [--no-char] \
        [--val-protocol shared|split] [--quiet]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GroupShuffleSplit

from src.improved_model_1.config import (
    ARTIFACT_DIR, C_ISSUE_GRID, C_TEAM_GRID, CHAR_MAX_FEATURES,
    MAX_ARTIFACT_MB, MIN_ISSUE_ROWS_ISOTONIC, USE_CHAR_DEFAULT,
    WORD_MAX_FEATURES,
)
from src.improved_model_1.features import build_vectorizer
from src.improved_model_1.model import HierarchicalSVM
from src.utils import config as C
from src.utils.data import load_holdout_2026, load_split
from src.utils.device import gpu_info
from src.utils.metrics import evaluate
from src.utils.registry import log_result
from src.utils.routing import (
    apply_per_team_thresholds, expected_calibration_error, north_star_summary,
    per_team_thresholds, predict_with_confidence,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-set", choices=("cap_issue", "3k_team"),
                   default="cap_issue")
    p.add_argument("--no-char", action="store_true")
    p.add_argument("--val-protocol", choices=("shared", "split"),
                   default="shared")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()
```

Commit the scaffold before implementing the body — makes review easier.

- [ ] **Step 2: Implement `_load_all_splits(train_set)`**

```python
def _load_all_splits(train_set: str) -> dict[str, pd.DataFrame]:
    train_name = "train_cap_issue" if train_set == "cap_issue" else "train_3k_team"
    return {
        "train": load_split(train_name),
        "val": load_split("val"),
        "test": load_split("test"),
        "holdout_2026": load_holdout_2026(),
    }
```

- [ ] **Step 3: Implement `_tune(...)` — Cs picked on val**

```python
def _tune(vec, train, val, use_char: bool) -> tuple[float, float, dict]:
    X_train = vec.fit_transform(train[C.TEXT_COL].astype(str))
    X_val = vec.transform(val[C.TEXT_COL].astype(str))
    y_train_t = train[C.TEAM_COL].to_numpy()
    y_val_t = val[C.TEAM_COL].to_numpy()
    y_train_i = train[C.ISSUE_COL].to_numpy()
    y_val_i = val[C.ISSUE_COL].to_numpy()

    grid_team = []
    for C_t in C_TEAM_GRID:
        m = HierarchicalSVM(C_team=C_t, C_issue=1.0)
        m.fit(X_train, y_train_t, y_train_i)
        pred = m.predict_team(X_val)
        f1 = f1_score(y_val_t, pred, average="macro")
        grid_team.append({"C_team": C_t, "val_team_macro_f1": f1})
    best_team = max(grid_team, key=lambda r: r["val_team_macro_f1"])

    grid_issue = []
    for C_i in C_ISSUE_GRID:
        m = HierarchicalSVM(C_team=best_team["C_team"], C_issue=C_i)
        m.fit(X_train, y_train_t, y_train_i)
        # ORACLE team: give the true team so we tune the issue head purely
        top = m.predict_issue_topk(X_val, teams=y_val_t, k=1)
        pred_i = np.array([r[0][0] for r in top])
        f1 = f1_score(y_val_i, pred_i, average="macro")
        grid_issue.append({"C_issue": C_i, "val_issue_macro_f1_oracle": f1})
    best_issue = max(grid_issue, key=lambda r: r["val_issue_macro_f1_oracle"])

    return best_team["C_team"], best_issue["C_issue"], {
        "team_grid": grid_team, "issue_grid": grid_issue,
        "best_team": best_team, "best_issue": best_issue,
    }
```

- [ ] **Step 4: Implement `_calibrate_and_route(...)`**

Handles both `--val-protocol shared` (fit calibration and thresholds on the
whole validation set — matches baseline 2) and `--val-protocol split` (group-aware
50/50 split of validation → calibrate on half A, choose thresholds on half B):

```python
def _split_val_for_protocol(val: pd.DataFrame, seed: int
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    idx_a, idx_b = next(gss.split(val, groups=val[C.GROUP_COL]))
    return val.iloc[idx_a].reset_index(drop=True), val.iloc[idx_b].reset_index(drop=True)


def _calibrate_and_route(model, vec, val, test, protocol: str, names):
    if protocol == "shared":
        X_val = vec.transform(val[C.TEXT_COL].astype(str))
        model.fit_team_calibrator(X_val, val[C.TEAM_COL].to_numpy())
        model.fit_issue_calibrators(X_val, val[C.TEAM_COL].to_numpy(),
                                    val[C.ISSUE_COL].to_numpy(),
                                    min_rows=MIN_ISSUE_ROWS_ISOTONIC)
        val_for_thresholds = val
        X_val_thr = X_val
    else:
        val_a, val_b = _split_val_for_protocol(val, C.RANDOM_SEED)
        X_val_a = vec.transform(val_a[C.TEXT_COL].astype(str))
        X_val_thr = vec.transform(val_b[C.TEXT_COL].astype(str))
        model.fit_team_calibrator(X_val_a, val_a[C.TEAM_COL].to_numpy())
        model.fit_issue_calibrators(X_val_a, val_a[C.TEAM_COL].to_numpy(),
                                    val_a[C.ISSUE_COL].to_numpy(),
                                    min_rows=MIN_ISSUE_ROWS_ISOTONIC)
        val_for_thresholds = val_b

    X_test = vec.transform(test[C.TEXT_COL].astype(str))
    val_proba = model.team_proba(X_val_thr)
    test_proba = model.team_proba(X_test)
    val_classes = model.calibrated_team_.classes_
    val_pred = val_classes[val_proba.argmax(1)]
    val_conf = val_proba.max(1)
    test_pred = val_classes[test_proba.argmax(1)]
    test_conf = test_proba.max(1)

    val_correct = val_pred == val_for_thresholds[C.TEAM_COL].to_numpy()
    test_correct = test_pred == test[C.TEAM_COL].to_numpy()
    ns, curves, teams = north_star_summary(
        val_conf, val_correct, test_conf, test_correct,
        test[C.TEAM_COL].to_numpy(), test_pred, names)
    per_team_thr = per_team_thresholds(val_conf, val_pred,
                                       val_for_thresholds[C.TEAM_COL].to_numpy(),
                                       target=0.95)
    routing_ptt = apply_per_team_thresholds(
        test[C.TEAM_COL].to_numpy(), test_pred, test_conf, per_team_thr,
        names=names)
    return {
        "ns": ns, "curves": curves, "teams": teams,
        "per_team_thresholds": per_team_thr,
        "per_team_routing_ptt": routing_ptt,
        "test_pred": test_pred, "test_conf": test_conf,
        "test_proba": test_proba, "X_test": X_test,
        "val_classes": val_classes,
    }
```

- [ ] **Step 5: Implement `_evaluate_and_write(...)`**

Compute team + issue metrics via `evaluate(...)`, write every artifact in the
baseline-2-compatible format:
- `metrics.json` (dummy + model on val/test),
- `classification_report.json`, `confusion_matrix.csv`,
- `north_star.json`, `coverage_curve.csv`, `per_team_routing.csv`,
- `predictions_test.csv` with columns
  `complaint_id, actual_team_id, actual_team_name, predicted_team_id,
   confidence, auto_routed, auto_routed_per_team,
   issue_top1, issue_top2, issue_top3, actual_issue_id`,
- `classification_report_issue.json`, `issue_metrics.json`,
- `tuning.json`, `run_info.json` (versions, `gpu_info()`, `train_set`,
  `C_team`, `C_issue`, row counts, wall-clock, artifact size, thresholds),
- `model.joblib` (with `compress=3`, after `model.compact()`).

Assert size:
```python
size_mb = (ARTIFACT_DIR / "model.joblib").stat().st_size / 1024**2
if size_mb > MAX_ARTIFACT_MB:
    raise RuntimeError(f"model.joblib is {size_mb:.1f} MB > {MAX_ARTIFACT_MB} MB budget")
```

- [ ] **Step 6: Implement `_evaluate_reference_models(...)`** for the registry

Trains four extra models with the same splits so `reports/model_comparison.csv`
can be compared apples-to-apples:

```python
def _evaluate_reference_models(train, val, test, holdout, names) -> None:
    # 1) flat_issue_ablation
    vec = build_vectorizer(use_char=False)  # word-only to keep memory sane
    X_train = vec.fit_transform(train[C.TEXT_COL].astype(str))
    X_val = vec.transform(val[C.TEXT_COL].astype(str))
    X_test = vec.transform(test[C.TEXT_COL].astype(str))
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    flat = LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=5000)
    flat.fit(X_train, train[C.ISSUE_COL])
    # Team is the mapping of the predicted issue
    from src.utils.labels import issue_to_team
    pred_issue = flat.predict(X_test)
    pred_team = np.array([issue_to_team[i] for i in pred_issue])
    log_result("flat_issue_ablation", "test", {
        "train_set": "cap_issue",
        "team_macro_f1": f1_score(test[C.TEAM_COL], pred_team, average="macro"),
        "issue_macro_f1": f1_score(test[C.ISSUE_COL], pred_issue, average="macro"),
        "n_eval": len(test),
    })

    # 2) nb_reference_cap_issue: recreate baseline 2 recipe on cap_issue
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.utils.class_weight import compute_sample_weight
    from src.utils.text import clean_text
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                   min_df=2, max_df=0.98, max_features=75_000,
                                   sublinear_tf=True, strip_accents="unicode")),
        ("clf", MultinomialNB(alpha=0.1, fit_prior=False)),
    ])
    sw = compute_sample_weight(class_weight="balanced", y=train[C.TEAM_COL])
    pipe.fit(train[C.TEXT_COL].astype(str), train[C.TEAM_COL],
             clf__sample_weight=sw)
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
    cal.fit(val[C.TEXT_COL].astype(str), val[C.TEAM_COL])
    pred = cal.predict(test[C.TEXT_COL].astype(str))
    log_result("nb_reference_cap_issue", "test", {
        "train_set": "cap_issue",
        "team_macro_f1": f1_score(test[C.TEAM_COL], pred, average="macro"),
        "n_eval": len(test),
    })

    # 3) dummy
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train[[C.TEXT_COL]], train[C.TEAM_COL])
    dpred = dummy.predict(test[[C.TEXT_COL]])
    log_result("dummy_most_frequent", "test", {
        "train_set": "cap_issue",
        "team_macro_f1": f1_score(test[C.TEAM_COL], dpred, average="macro"),
        "n_eval": len(test),
    })

    # 4) baseline_model_2 on the 2026 cohort (load its committed model)
    from src.utils.config import BASELINE2_METRICS
    b2 = joblib.load(BASELINE2_METRICS / "model.joblib")
    hpred = b2.predict(holdout[C.TEXT_COL].astype(str))
    log_result("baseline_model_2", "holdout_2026", {
        "train_set": "3k_team",
        "team_macro_f1": f1_score(holdout[C.TEAM_COL], hpred, average="macro"),
        "n_eval": len(holdout),
    })
```

- [ ] **Step 7: `main()` orchestration**

```python
def main() -> None:
    args = _parse_args()
    verbose = not args.quiet
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    splits = _load_all_splits(args.train_set)
    train, val, test, holdout = (splits["train"], splits["val"],
                                 splits["test"], splits["holdout_2026"])
    names = dict(zip(pd.concat([train, val, test])[C.TEAM_COL],
                     pd.concat([train, val, test])["team_name"]))

    vec = build_vectorizer(
        use_char=(not args.no_char) and USE_CHAR_DEFAULT,
        word_max=WORD_MAX_FEATURES, char_max=CHAR_MAX_FEATURES)

    C_t, C_i, tuning = _tune(vec, train, val, use_char=not args.no_char)
    (ARTIFACT_DIR / "tuning.json").write_text(json.dumps(tuning, indent=2))

    # Refit on full train with best Cs
    X_train = vec.fit_transform(train[C.TEXT_COL].astype(str))
    model = HierarchicalSVM(C_team=C_t, C_issue=C_i)
    model.fit(X_train, train[C.TEAM_COL].to_numpy(),
              train[C.ISSUE_COL].to_numpy())

    routed = _calibrate_and_route(model, vec, val, test,
                                  protocol=args.val_protocol, names=names)
    _evaluate_and_write(model, vec, val, test, holdout, routed,
                        args=args, tuning=tuning, wall_clock=time.time() - t0)
    _evaluate_reference_models(train, val, test, holdout, names)
```

- [ ] **Step 8: Run the primary training run**

```bash
cd F:/NLP_MSML641 && F:/NLP_MSML641/.venv/Scripts/python -m src.improved_model_1.train
```
Expected: prints per-class table, north-star line, artifacts in
`docs/metrics/improved_model_1/`, `model.joblib` size ≤ 95 MB, macro-F1 on
test > 0.647 (baseline 2 target). If macro-F1 ≤ 0.647, STOP and report — this
is an acceptance-criterion failure.

- [ ] **Step 9: Repeat for `--train-set 3k_team`**

```bash
F:/NLP_MSML641/.venv/Scripts/python -m src.improved_model_1.train --train-set 3k_team
```
Rename or copy artifacts under `docs/metrics/improved_model_1_3k/` OR call
`log_result("improved_model_1_3k", "test", ...)` directly. Recommendation: pass
`--train-set 3k_team` and have `train.py` write to
`METRICS_ROOT / "improved_model_1_3k"` when `args.train_set == "3k_team"`
(update `ARTIFACT_DIR` at run time). Adjust code accordingly in Step 5.

- [ ] **Step 10: Repeat for `--val-protocol split`**

```bash
F:/NLP_MSML641/.venv/Scripts/python -m src.improved_model_1.train --val-protocol split
```
Write to `docs/metrics/improved_model_1_valsplit/`.

- [ ] **Step 11: Commit code + all metric artifacts (NOT `model.joblib` yet)**

```bash
cd F:/NLP_MSML641 && git add src/improved_model_1/train.py \
    docs/metrics/improved_model_1/metrics.json \
    docs/metrics/improved_model_1/classification_report.json \
    docs/metrics/improved_model_1/confusion_matrix.csv \
    docs/metrics/improved_model_1/north_star.json \
    docs/metrics/improved_model_1/per_team_routing.csv \
    docs/metrics/improved_model_1/coverage_curve.csv \
    docs/metrics/improved_model_1/predictions_test.csv \
    docs/metrics/improved_model_1/classification_report_issue.json \
    docs/metrics/improved_model_1/issue_metrics.json \
    docs/metrics/improved_model_1/tuning.json \
    docs/metrics/improved_model_1/run_info.json \
    docs/metrics/improved_model_1_3k/ \
    docs/metrics/improved_model_1_valsplit/ \
    reports/model_comparison.csv
git commit -m "feat(improved_model_1): training pipeline + evaluation artifacts"
```

Note: `model.joblib` stays **unstaged** until §5 approves the app swap.

---

### Task 6: `src/improved_model_1/predict.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_improved_model_1_predict.py`:
```python
"""Predict output contract test. Runs only if model.joblib exists."""
from pathlib import Path
import pytest

from src.improved_model_1.config import ARTIFACT_DIR


@pytest.mark.skipif(not (ARTIFACT_DIR / "model.joblib").exists(),
                    reason="model.joblib not yet built")
def test_predict_output_shape():
    from src.improved_model_1.predict import predict
    out = predict("A collector keeps calling about a debt I already paid.")
    required = {"predicted_team_id", "predicted_team_name", "confidence",
                "top_k", "issue_top_k", "route", "model_name"}
    assert required.issubset(out)
    assert out["route"] in ("auto", "review")
    assert out["model_name"] == "improved_model_1"
```

- [ ] **Step 2: Implement `predict.py`**

Signature superset of `src.baseline_model_2.predict.predict()`. Loads
`model.joblib`, calls team classifier, then issue classifier of predicted team,
applies per-team threshold (falling back to global) saved in the artifact for
`route`. CLI: `python -m src.improved_model_1.predict "<text>"`.

- [ ] **Step 3: Verify**

Run: `pytest tests/test_improved_model_1_predict.py -q`
Expected: passes if `model.joblib` exists, skipped otherwise.

- [ ] **Step 4: Commit**

```bash
cd F:/NLP_MSML641 && git add src/improved_model_1/predict.py \
    tests/test_improved_model_1_predict.py
git commit -m "feat(improved_model_1): predict() CLI + importable API"
```

---

### Task 7: Write `docs/improved_model_1.md`

- [ ] **Step 1: Draft the document**

Structure follows `docs/baseline_model_2.md`:
- Objective and design
- Data (reuses §1 split numbers; link to `split_report.json`)
- The model (vectorizer, hierarchy, calibration, thresholds)
- Evaluation table vs baseline 2 on the identical test set (numbers from
  `metrics.json`, `north_star.json`, `per_team_routing.csv`)
- Per-team table
- North star: global and per-team thresholds with 95% CIs
- Guardrail
- Issue results (macro-F1, top-3, oracle ceiling)
- 3k-vs-cap comparison (from `improved_model_1_3k/` artifacts)
- Hard pairs (T02↔T01, T05↔T04, T11↔T01/T02)
- 2026 cohort forward-drift check
- Known limitations
- How to run + generated artifacts

- [ ] **Step 2: Add a short section to top-level `README.md`**

Insert an "Improved Model 1" subsection under the Baseline 2 section. Do **not**
change baseline 2's text.

- [ ] **Step 3: Commit**

```bash
cd F:/NLP_MSML641 && git add docs/improved_model_1.md README.md
git commit -m "docs: improved_model_1 write-up + top-level README section"
```

---

### Task 8: Push branch and report the comparison table

- [ ] **Step 1: Push**

```bash
cd F:/NLP_MSML641 && git push -u origin improved_model_1
```

- [ ] **Step 2: Print the comparison table**

Run:
```bash
F:/NLP_MSML641/.venv/Scripts/python -c "import pandas as pd; \
    df = pd.read_csv('F:/NLP_MSML641/reports/model_comparison.csv'); \
    print(df.to_string(index=False))"
```

- [ ] **Step 3: STOP and ask user**

Ask:
> "Improved model 1 outperforms baseline 2 on macro-F1 by X points on the
> identical test set. Do you want to (a) integrate it into the app (plan
> `04-app-integration.md`) or (b) continue to model 2 (plan
> `03-improved-model-2-embeddings.md`)?"

---

## Acceptance criteria (verbatim from source spec §2.8)

- [ ] Team macro-F1 on test > 0.647.
- [ ] North-star coverage at the validation-fit 95% threshold ≥ 46.5% (test
  precision and CI reported honestly, even if the target isn't met exactly).
- [ ] Per-team guardrail: every team that auto-routes stays ≥ 0.85 routed
  precision on test, OR the breach is explicitly documented.
- [ ] Hierarchical issue macro-F1 ≥ flat ablation.
- [ ] `model.joblib` ≤ 95 MB. `predict()` output is a superset of baseline 2's.
- [ ] `reports/model_comparison.csv` contains rows for
  `baseline_model_2` (seed), `dummy_most_frequent`, `nb_reference_cap_issue`,
  `improved_model_1`, `improved_model_1_3k`, `flat_issue_ablation` on test,
  plus `baseline_model_2` and `improved_model_1` on `holdout_2026`.

## Self-review checklist

- [ ] Types match across tasks: `HierarchicalSVM` is referenced consistently
  (Tasks 4, 5, 6). `predict_issue_topk` returns
  `list[list[tuple[str, float]]]` everywhere.
- [ ] No placeholders in code steps — every step shows the actual code.
- [ ] Test files under `tests/` mirror source paths.
- [ ] `model.joblib` is explicitly excluded from Task 5 commit.
- [ ] `pytest -m 'not slow'` still passes throughout (regression check).

## Handoff

Depending on user decision:
- (a) app integration → plan `04-app-integration.md`
- (b) model 2 embeddings → plan `03-improved-model-2-embeddings.md`
