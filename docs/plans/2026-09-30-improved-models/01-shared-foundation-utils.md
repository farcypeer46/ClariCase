# 01 — Shared Foundation (`src/utils/`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared library used by every improved model — config,
labels, text preprocessing, temporal splits, metrics, north-star routing, and a
results registry — and prove parity with `baseline_model_2` on the identical
test set before any new model is trained.

**Architecture:** Read-only importable package `src/utils/` mirroring
`src/baseline_model_2/`'s protocol. Splits are cached to
`data/cache/splits_v2/` (gitignored) as Parquet. Metrics/routing helpers are
copied (not imported) from baseline 2 so the served model's code path is never
touched. A single parity test locks in the guarantee that our re-implementation
matches baseline 2's committed numbers.

**Tech Stack:** pandas, numpy, scikit-learn 1.9.1, pyarrow, joblib, pytest.

**Prerequisite:** plan `00-environment-and-repo-hygiene.md` complete; branch
`improved_model_1` checked out; `data/cache/` gitignored.

---

## File layout produced

```
src/utils/
├── __init__.py
├── config.py
├── labels.py
├── text.py
├── build_splits.py
├── data.py
├── device.py
├── metrics.py
├── routing.py
└── registry.py
tests/
├── __init__.py
└── test_parity_baseline2.py
docs/metrics/improved_model_1/
├── split_report.json           # copied from data/cache/splits_v2/split_report.json
└── issue_canonical.csv         # copied from data/processed/issue_canonical.csv
reports/
└── model_comparison.csv        # created with baseline_2 seed row
data/cache/splits_v2/           # gitignored
├── train_cap_issue.parquet
├── train_3k_team.parquet
├── val.parquet
├── test.parquet
├── train_full_ids.parquet
└── split_report.json
```

`src/utils/serving.py` is **NOT** created here — it belongs to the
app-integration plan (04).

---

### Task 1: Package skeleton

**Files:**
- Create: `src/utils/__init__.py` (empty except a version string)
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Create `src/utils/__init__.py`**

Write:
```python
"""Shared utilities for improved_model_N. See docs/plans/2026-09-30-improved-models/."""

__all__ = []
```

- [ ] **Step 2: Create `tests/__init__.py`**

Write an empty file.

- [ ] **Step 3: Commit**

```bash
cd F:/NLP_MSML641 && git add src/utils/__init__.py tests/__init__.py
git commit -m "feat(utils): package skeleton for improved models"
```

---

### Task 2: `src/utils/config.py`

**Files:**
- Create: `src/utils/config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_utils_config.py`:
```python
from pathlib import Path
from src.utils import config as c


def test_paths_resolve_under_repo_root():
    assert c.ROOT.is_dir()
    assert c.FULL_CSV.name == "complaints_product_issues_2024_2025.csv"
    assert c.SPLITS_V2 == c.CACHE / "splits_v2"
    assert c.BASELINE2_METRICS == c.ROOT / "docs" / "metrics" / "baseline_model_2"


def test_temporal_boundaries_match_baseline_2():
    assert (c.TRAIN_END, c.VAL_END, c.TEST_END) == (
        "2025-07-01", "2025-10-01", "2026-01-01")


def test_seed_and_caps():
    assert c.RANDOM_SEED == 42
    assert c.TRAIN_CAP_PER_ISSUE == 10_000
    assert c.BASELINE2_CAP_PER_TEAM == 3_000
    assert c.MERGE_ISSUE_VARIANTS is True
    assert c.ISSUE_VARIANT_MERGES["T01_I007"] == "T01_I008"
    assert c.ISSUE_VARIANT_MERGES["T01_I003"] == "T01_I001"


def test_routing_constants_match_baseline_2():
    assert c.PRECISION_TARGETS == (0.95, 0.90, 0.85)
    assert c.TEAM_PRECISION_FLOOR == 0.85
    assert c.MIN_ROUTED == 50
    assert c.N_BOOTSTRAP == 1000
    assert c.ECE_BINS == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_utils_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.config'`.

- [ ] **Step 3: Write `src/utils/config.py`**

Verbatim from spec §1.1 (copy the whole snippet). One import block at the top:
```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FULL_CSV = PROCESSED / "complaints_product_issues_2024_2025.csv"
CACHE = ROOT / "data" / "cache"
FULL_PARQUET = CACHE / "complaints_product_issues_2024_2025.parquet"
SPLITS_V2 = CACHE / "splits_v2"
HOLDOUT_ZIP = ROOT / "data" / "complaint_router_2026_holdout.zip"
HOLDOUT_PARQUET = CACHE / "holdout_2026" / "fresh_holdout_2026_01_02.parquet"
ISSUE_MAPPING = PROCESSED / "issue_mapping.csv"
TEAM_MAPPING = PROCESSED / "team_mapping.csv"
BASELINE2_METRICS = ROOT / "docs" / "metrics" / "baseline_model_2"
METRICS_ROOT = ROOT / "docs" / "metrics"
REPORTS = ROOT / "reports"

TEXT_COL, TEAM_COL, ISSUE_COL = "complaint_text", "team_id", "issue_id"
GROUP_COL, DATE_COL, ID_COL = "leakage_group_id", "date_received", "complaint_id"
USECOLS = [ID_COL, DATE_COL, "company", TEXT_COL, TEAM_COL, "team_name",
           "product_issue_id", GROUP_COL]

RANDOM_SEED = 42
TRAIN_END = "2025-07-01"
VAL_END = "2025-10-01"
TEST_END = "2026-01-01"
TRAIN_CAP_PER_ISSUE = 10_000
BASELINE2_CAP_PER_TEAM = 3_000
MERGE_ISSUE_VARIANTS = True
ISSUE_VARIANT_MERGES = {"T01_I007": "T01_I008",
                        "T01_I003": "T01_I001"}

PRECISION_TARGETS = (0.95, 0.90, 0.85)
TEAM_PRECISION_FLOOR = 0.85
MIN_ROUTED = 50
N_BOOTSTRAP = 1000
ECE_BINS = 10
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_utils_config.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd F:/NLP_MSML641 && git add src/utils/config.py tests/test_utils_config.py
git commit -m "feat(utils): config constants (paths, seed, temporal splits, routing)"
```

---

### Task 3: `src/utils/text.py`

**Files:**
- Create: `src/utils/text.py`
- Create: `tests/test_utils_text.py`

- [ ] **Step 1: Write the failing test**

```python
from src.utils.text import clean_text, tokenize


def test_reexports_advanced_clean_text():
    from data.preparation.advanced_text import clean_text as canonical
    assert clean_text is canonical  # exact reference, not a copy


def test_tokenize_lowercases_and_strips_numbers():
    tokens = tokenize("Call 1234567 don't ignore me!!")
    assert "don't" in tokens
    assert "call" in tokens
    # 6+ digit numbers are redacted to "number" by advanced clean_text
    assert "1234567" not in tokens
    assert "number" in tokens
```

- [ ] **Step 2: Run to see it fail**

Run: `pytest tests/test_utils_text.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Write `src/utils/text.py`:
```python
"""Canonical text preprocessing and tokenization for improved models.

`clean_text` is imported (not copied) from `data.preparation.advanced_text`
so pickled TfidfVectorizers reference the stable canonical function.
"""
import re

from data.preparation.advanced_text import clean_text

__all__ = ["clean_text", "tokenize", "TOKEN_RE"]

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(clean_text(text))
```

- [ ] **Step 4: Verify tests pass**

Run: `pytest tests/test_utils_text.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd F:/NLP_MSML641 && git add src/utils/text.py tests/test_utils_text.py
git commit -m "feat(utils): re-export clean_text and add tokenize()"
```

---

### Task 4: `src/utils/labels.py`

**Files:**
- Create: `src/utils/labels.py`
- Create: `tests/test_utils_labels.py`
- Create (via first run): `data/processed/issue_canonical.csv`
- Create (via first run): `docs/metrics/improved_model_1/issue_canonical.csv`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from src.utils import labels as L
from src.utils.config import ISSUE_VARIANT_MERGES


def test_canonical_issue_applies_merges():
    for src, dst in ISSUE_VARIANT_MERGES.items():
        assert L.canonical_issue(src) == dst
    assert L.canonical_issue("T02_I001") == "T02_I001"


def test_issue_to_team_and_reverse_are_consistent():
    for iid, tid in L.issue_to_team.items():
        assert iid in L.team_to_issues[tid]
    for tid, iids in L.team_to_issues.items():
        assert all(L.issue_to_team[i] == tid for i in iids)


def test_team_issue_mask_shape_and_diagonal():
    team_classes = sorted(L.team_to_issues)
    issue_classes = sorted(L.issue_to_team)
    mask = L.team_issue_mask(issue_classes, team_classes)
    assert mask.shape == (len(team_classes), len(issue_classes))
    for i, iid in enumerate(issue_classes):
        t_row = team_classes.index(L.issue_to_team[iid])
        assert mask[t_row, i]
```

- [ ] **Step 2: Run to see it fail**

Run: `pytest tests/test_utils_labels.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Write `src/utils/labels.py`:
```python
"""Team and issue label mappings, plus canonicalization for merged issues."""
from __future__ import annotations

import pandas as pd
import numpy as np

from src.utils.config import (
    ISSUE_MAPPING, ISSUE_VARIANT_MERGES, MERGE_ISSUE_VARIANTS,
    METRICS_ROOT, PROCESSED, TEAM_MAPPING,
)


def load_issue_mapping() -> pd.DataFrame:
    return pd.read_csv(ISSUE_MAPPING)


def canonical_issue(product_issue_id: str) -> str:
    if not MERGE_ISSUE_VARIANTS:
        return product_issue_id
    return ISSUE_VARIANT_MERGES.get(product_issue_id, product_issue_id)


def _build_tables() -> tuple[pd.DataFrame, dict, dict, dict]:
    df = load_issue_mapping().copy()
    # Expected columns: product_issue_id, team_id, issue_label (verify at read).
    for col in ("product_issue_id", "team_id"):
        if col not in df.columns:
            raise KeyError(
                f"issue_mapping.csv missing '{col}'; got {df.columns.tolist()}")
    df["issue_id"] = df["product_issue_id"].map(canonical_issue)

    canonical_cols = ["product_issue_id", "issue_id", "team_id"]
    if "issue_label" in df.columns:
        canonical_cols.append("issue_label")
    out = df[canonical_cols].drop_duplicates()

    issue_to_team = dict(zip(out["issue_id"], out["team_id"]))
    team_to_issues: dict[str, list[str]] = {}
    for tid, sub in out.groupby("team_id"):
        team_to_issues[tid] = sorted(sub["issue_id"].unique())

    team_df = pd.read_csv(TEAM_MAPPING)
    team_names = dict(zip(team_df["team_id"], team_df["team_name"]))
    return out, issue_to_team, team_to_issues, team_names


_canonical_df, issue_to_team, team_to_issues, team_names = _build_tables()


def team_issue_mask(issue_classes: list[str],
                    team_classes: list[str]) -> np.ndarray:
    mask = np.zeros((len(team_classes), len(issue_classes)), dtype=bool)
    t_index = {t: i for i, t in enumerate(team_classes)}
    for j, iid in enumerate(issue_classes):
        t = issue_to_team.get(iid)
        if t in t_index:
            mask[t_index[t], j] = True
    return mask


def write_canonical_csv() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    _canonical_df.to_csv(PROCESSED / "issue_canonical.csv", index=False)
    tracked = METRICS_ROOT / "improved_model_1"
    tracked.mkdir(parents=True, exist_ok=True)
    _canonical_df.to_csv(tracked / "issue_canonical.csv", index=False)


if __name__ == "__main__":
    write_canonical_csv()
    print(f"[labels] {len(_canonical_df)} issues; "
          f"{len(team_to_issues)} teams; canonical CSV written")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_utils_labels.py -q`
Expected: 3 passed.

- [ ] **Step 5: Generate the canonical CSVs**

Run: `F:/NLP_MSML641/.venv/Scripts/python -m src.utils.labels`
Expected: prints the summary line; both CSVs now exist.

- [ ] **Step 6: Commit**

```bash
cd F:/NLP_MSML641 && git add src/utils/labels.py tests/test_utils_labels.py \
    docs/metrics/improved_model_1/issue_canonical.csv
# data/processed/issue_canonical.csv is gitignored under data/processed/ — do NOT stage it
git commit -m "feat(utils): label mappings, canonicalization, team↔issue mask"
```

---

### Task 5: `src/utils/device.py`

**Files:**
- Create: `src/utils/device.py`
- Create: `tests/test_utils_device.py`

- [ ] **Step 1: Write the failing test**

```python
from src.utils.device import get_device, gpu_info, amp_dtype, suggested_batch_size


def test_get_device_returns_cpu_when_preferred():
    assert get_device(prefer="cpu") == "cpu"


def test_gpu_info_returns_dict_with_versions():
    info = gpu_info()
    assert set(info) >= {"torch", "cuda_available"}


def test_suggested_batch_size_returns_positive_int():
    assert suggested_batch_size(vram_gb=8, task="sbert") > 0
    assert suggested_batch_size(vram_gb=24, task="sbert") \
           >= suggested_batch_size(vram_gb=8, task="sbert")
```

- [ ] **Step 2: Run tests — expect fail**

Run: `pytest tests/test_utils_device.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Write `src/utils/device.py`:
```python
"""Device selection and GPU introspection. torch imported lazily so the
Streamlit app (which doesn't ship torch) can still import src.utils.*.
"""
from __future__ import annotations


def _torch():
    import torch  # noqa: WPS433 (lazy on purpose)
    return torch


def get_device(prefer: str = "auto") -> str:
    if prefer == "cpu":
        return "cpu"
    try:
        torch = _torch()
    except ImportError:
        return "cpu"
    if prefer == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_info() -> dict:
    try:
        torch = _torch()
    except ImportError:
        return {"torch": None, "cuda_available": False}
    info = {"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["device_name"] = props.name
        info["total_vram_gb"] = round(props.total_memory / 1024**3, 2)
        info["cuda_version"] = torch.version.cuda
    return info


def amp_dtype():
    try:
        torch = _torch()
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# vram_gb bucket -> {task -> batch_size}. Uses the table in spec §0.1.
_TABLE = {
    "sbert":         {8: 64,  16: 128, 24: 256},
    "bert_base":     {8: 8,   16: 16,  24: 32},
    "modernbert":    {8: 4,   16: 8,   24: 16},
}


def suggested_batch_size(vram_gb: float, task: str) -> int:
    bucket = 8 if vram_gb <= 8 else (16 if vram_gb <= 16 else 24)
    return _TABLE.get(task, _TABLE["sbert"])[bucket]
```

- [ ] **Step 4: Verify**

Run: `pytest tests/test_utils_device.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd F:/NLP_MSML641 && git add src/utils/device.py tests/test_utils_device.py
git commit -m "feat(utils): device selection, gpu_info(), amp_dtype()"
```

---

### Task 6: `src/utils/build_splits.py` — CSV → Parquet + temporal splits

**Files:**
- Create: `src/utils/build_splits.py`
- Create: `tests/test_build_splits.py`

- [ ] **Step 1: Write the failing counts test**

```python
"""End-to-end split builder test. Slow: reads 1.9 GB CSV once, then Parquet."""
import json
import pandas as pd
import pytest

from src.utils import config as c


@pytest.mark.slow
def test_splits_match_official_partitions():
    from src.utils.build_splits import build_all
    build_all(verbose=False)
    report = json.loads((c.SPLITS_V2 / "split_report.json").read_text())
    counts = report["counts"]
    assert counts["train_full"] == 774_412
    assert counts["val"] == 153_502
    assert counts["test"] == 109_834
    assert report["dropped_crossing_group_rows"] == 6_867
    assert report["dropped_crossing_groups"] == 1_949


@pytest.mark.slow
def test_test_ids_equal_baseline2():
    b2 = pd.read_csv(c.BASELINE2_METRICS / "predictions_test.csv",
                     usecols=[c.ID_COL], dtype={c.ID_COL: "string"})
    ours = pd.read_parquet(c.SPLITS_V2 / "test.parquet", columns=[c.ID_COL])
    assert set(ours[c.ID_COL]) == set(b2[c.ID_COL])
```

Mark slow tests via `pytest.ini` (create at repo root if missing):
```
[pytest]
markers =
    slow: expensive tests that read the 1.9 GB CSV; run with `pytest -m slow`
```

- [ ] **Step 2: Run only the fast tests to confirm no regression**

Run: `pytest -q -m 'not slow'`
Expected: all previously-passing tests still pass.

- [ ] **Step 3: Implement `build_splits.py`**

Write `src/utils/build_splits.py`. Structure:

1. `csv_to_parquet()`: if `FULL_PARQUET` missing, stream CSV with
   `chunksize=100_000, usecols=USECOLS, dtype={ID_COL: "string"},
   keep_default_na=False`, append to a `pyarrow.parquet.ParquetWriter`.
2. `assign_periods(df)`: `date_received` → `train`/`val`/`test`/`""` via
   `np.select` on `TRAIN_END`, `VAL_END`, `TEST_END`.
3. `drop_crossing_groups(df)`: drop every row whose `leakage_group_id` appears
   in more than one non-empty period; record dropped row and group counts.
4. `derive_training_subsets(train_df)`:
   - `train_cap_issue` = `train_df.groupby('issue_id').sample(n=TRAIN_CAP_PER_ISSUE,
     random_state=RANDOM_SEED, replace=False)` with a fallback to all rows when
     a group is smaller than the cap. Use
     `train_df.groupby('issue_id', group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), TRAIN_CAP_PER_ISSUE), random_state=RANDOM_SEED))`.
   - `train_3k_team` = `train_df.sample(frac=1, random_state=RANDOM_SEED)
     .groupby('team_id').head(BASELINE2_CAP_PER_TEAM)`.
   - `train_full_ids` = `train_df[[ID_COL]]`.
5. `assert_invariants(train, val, test)`:
   - group sets pairwise disjoint;
   - all 11 teams present in each split;
   - `len(val) == 153_502`, `len(test) == 109_834`;
   - test IDs equal baseline 2's `predictions_test.csv` IDs (assert set
     equality; print differences and raise on mismatch).
6. `write_all(...)`: writes the five Parquets and `split_report.json` with:
   ```json
   {"counts": {"train_full": ..., "train_cap_issue": ..., "train_3k_team": ...,
    "val": ..., "test": ...},
    "per_team": {...}, "per_issue": {...},
    "dropped_crossing_group_rows": ..., "dropped_crossing_groups": ...,
    "date_ranges": {...}, "parity": {"test_ids_equal_baseline2": true}}
   ```
7. `build_all(verbose=True)`: orchestrator.
8. Also copy the report to `docs/metrics/improved_model_1/split_report.json`.

Ensure schema: `complaint_id, date_received, company, complaint_text, team_id,
team_name, issue_id, leakage_group_id`.

Add a `__main__` block: `python -m src.utils.build_splits`.

- [ ] **Step 4: Run once against the real CSV**

Run:
```bash
F:/NLP_MSML641/.venv/Scripts/python -m src.utils.build_splits
```
Expected: prints counts matching README; writes to `data/cache/splits_v2/`.
Wall-clock 5–15 min the first time (Parquet materialization dominates).

- [ ] **Step 5: Run the slow tests**

Run: `pytest tests/test_build_splits.py -q -m slow`
Expected: 2 passed.

- [ ] **Step 6: Copy split_report.json into the tracked metrics folder**

Run:
```bash
cp F:/NLP_MSML641/data/cache/splits_v2/split_report.json \
   F:/NLP_MSML641/docs/metrics/improved_model_1/split_report.json
```

- [ ] **Step 7: Commit code + tracked report (not the Parquets)**

```bash
cd F:/NLP_MSML641 && git add src/utils/build_splits.py tests/test_build_splits.py \
    docs/metrics/improved_model_1/split_report.json pytest.ini
git commit -m "feat(utils): temporal + group-aware split builder with parity check"
```

If `pytest.ini` already exists and only needed the marker, stage only the diff.

---

### Task 7: `src/utils/data.py`

**Files:**
- Create: `src/utils/data.py`
- Create: `tests/test_utils_data.py`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
import pytest

from src.utils import data as D
from src.utils.config import ID_COL


@pytest.mark.slow
def test_load_split_returns_expected_columns():
    df = D.load_split("val")
    expected = {"complaint_id", "date_received", "company", "complaint_text",
                "team_id", "team_name", "issue_id", "leakage_group_id"}
    assert expected.issubset(df.columns)
    assert df[ID_COL].dtype == "string" or df[ID_COL].dtype == "object"


@pytest.mark.slow
def test_load_holdout_has_issue_id():
    df = D.load_holdout_2026()
    assert "issue_id" in df.columns
    assert len(df) == 29_798
```

- [ ] **Step 2: Run — expect fail**

Run: `pytest tests/test_utils_data.py -q -m slow`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Write `src/utils/data.py`:
```python
"""Reads persisted splits and the 2026 holdout cohort."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from src.utils.config import (
    HOLDOUT_PARQUET, HOLDOUT_ZIP, ID_COL, SPLITS_V2,
)
from src.utils.labels import canonical_issue


_VALID_SPLITS = {"train_cap_issue", "train_3k_team", "val", "test"}


def load_split(name: str) -> pd.DataFrame:
    if name not in _VALID_SPLITS:
        raise ValueError(f"Unknown split {name!r}; want one of {_VALID_SPLITS}")
    return pd.read_parquet(SPLITS_V2 / f"{name}.parquet")


def _extract_holdout() -> Path:
    HOLDOUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    if HOLDOUT_PARQUET.exists():
        return HOLDOUT_PARQUET
    with zipfile.ZipFile(HOLDOUT_ZIP) as zf:
        member = next(n for n in zf.namelist() if n.endswith("fresh_holdout_2026_01_02.parquet"))
        zf.extract(member, HOLDOUT_PARQUET.parent)
        extracted = HOLDOUT_PARQUET.parent / member
        if extracted != HOLDOUT_PARQUET:
            extracted.rename(HOLDOUT_PARQUET)
    return HOLDOUT_PARQUET


def load_holdout_2026() -> pd.DataFrame:
    path = _extract_holdout()
    df = pd.read_parquet(path)
    required = {"complaint_id", "complaint_text", "team_id", "product_issue_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"holdout missing {missing}; got {df.columns.tolist()}")
    df["issue_id"] = df["product_issue_id"].map(canonical_issue)
    return df
```

- [ ] **Step 4: Run tests to verify**

Run: `pytest tests/test_utils_data.py -q -m slow`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd F:/NLP_MSML641 && git add src/utils/data.py tests/test_utils_data.py
git commit -m "feat(utils): load_split() and load_holdout_2026()"
```

---

### Task 8: `src/utils/metrics.py`

**Files:**
- Create: `src/utils/metrics.py`
- Create: `tests/test_utils_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from src.utils.metrics import evaluate, bootstrap_ci_macro_f1


def _fake():
    y_true = np.array(["T01", "T01", "T02", "T02", "T03"])
    proba = np.array([[.7, .2, .1], [.3, .4, .3], [.1, .8, .1],
                      [.2, .6, .2], [.1, .2, .7]])
    return y_true, proba, np.array(["T01", "T02", "T03"])


def test_evaluate_team_only_returns_expected_keys():
    y, p, classes = _fake()
    out = evaluate(y_team=y, team_proba=p, team_classes=classes)
    assert set(out["team"]).issuperset({"accuracy", "macro_f1", "weighted_f1",
                                        "per_class", "confusion_matrix"})
    assert out.get("issue") is None


def test_bootstrap_ci_returns_lo_hi():
    y, p, classes = _fake()
    lo, hi = bootstrap_ci_macro_f1(y, classes[p.argmax(1)], n_boot=50, seed=0)
    assert 0 <= lo <= hi <= 1
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement `src/utils/metrics.py`**

Signature per spec §1.6:
```python
def evaluate(y_team, team_proba, team_classes,
             y_issue=None, issue_scores=None, issue_classes=None,
             oracle_issue_scores=None) -> dict
```
Sections:
- `team`: `accuracy`, `macro_f1`, `weighted_f1`, `per_class` (dict from
  `classification_report(output_dict=True)`), `confusion_matrix` (as
  `pd.DataFrame` → serialized to list of lists), `macro_f1_ci_95` from
  `bootstrap_ci_macro_f1(...)`.
- `issue`: same plus `top3_accuracy` when `issue_scores` and `y_issue` given.
- `joint`: fraction of rows where team-pred AND top-1-issue-pred both match.
- `oracle`: run issue metrics again with `oracle_issue_scores` (the per-team
  classifier restricted to the true team's issues).
- `hard_pairs`: dict with recall of T02/T05/T11 and share predicted as
  T01/T04/(T01|T02) respectively.

Helper `bootstrap_ci_macro_f1(y_true, y_pred, n_boot, seed)` uses
`np.random.default_rng`, resamples row indices with replacement, and returns
`(percentile 2.5, percentile 97.5)`.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_utils_metrics.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd F:/NLP_MSML641 && git add src/utils/metrics.py tests/test_utils_metrics.py
git commit -m "feat(utils): unified evaluate() for team, issue, joint, oracle, hard-pairs"
```

---

### Task 9: `src/utils/routing.py` — port baseline 2's north-star harness

**Files:**
- Create: `src/utils/routing.py`
- Create: `tests/test_utils_routing_synthetic.py`

- [ ] **Step 1: Copy the eight functions verbatim from
  `src/baseline_model_2/train.py`** into `src/utils/routing.py`:
  `expected_calibration_error`, `coverage_curve`, `threshold_for`,
  `routed_stats`, `bootstrap_ci`, `per_team_routing`, `north_star_summary`,
  and `predict_with_confidence`.

Header of the file:
```python
"""Routing / north-star harness.

Ported verbatim from `src/baseline_model_2/train.py` (see that file for the
original implementation; this copy exists so improved models can import the
harness without depending on the served baseline module).
"""
```

- [ ] **Step 2: Add `per_team_thresholds()` and `apply_per_team_thresholds()`**

Signature:
```python
def per_team_thresholds(val_conf: np.ndarray, val_pred: np.ndarray,
                        val_true: np.ndarray, target: float = 0.95,
                        min_routed: int = MIN_ROUTED) -> dict[str, float | None]:
    """For each predicted team, lowest threshold where routed-to-team precision
    on validation reaches `target`. None if unreachable (that team never
    auto-routes)."""
```
Iterate thresholds from `np.linspace(0, 1, 501)`; for each team, filter
`val_pred == team & val_conf >= t`, require `n_routed >= min_routed`, and
capture the smallest feasible `t`.

```python
def apply_per_team_thresholds(y_true, y_pred, conf, thresholds, names, min_routed=MIN_ROUTED):
    """Return {coverage, precision, per_team_df, bootstrap_ci} on the given
    split when each team uses its own threshold from `thresholds`."""
```

- [ ] **Step 3: Write a small synthetic test**

Create `tests/test_utils_routing_synthetic.py` — build a 20-row toy problem
with known answers, verify `coverage_curve` monotonicity in threshold,
`threshold_for(0.5) is not None`, and that `per_team_thresholds` returns None
for a team that never appears in `y_pred`.

- [ ] **Step 4: Verify**

Run: `pytest tests/test_utils_routing_synthetic.py -q`

- [ ] **Step 5: Commit**

```bash
cd F:/NLP_MSML641 && git add src/utils/routing.py tests/test_utils_routing_synthetic.py
git commit -m "feat(utils): port north-star harness + per-team thresholds"
```

---

### Task 10: `src/utils/registry.py` + baseline 2 seed row

**Files:**
- Create: `src/utils/registry.py`
- Create: `tests/test_utils_registry.py`
- Create: `reports/model_comparison.csv` (seeded with baseline_2)

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from pathlib import Path

from src.utils.registry import log_result, seed_from_baseline_2, COLUMNS
from src.utils.config import REPORTS


def test_log_result_appends_row(tmp_path, monkeypatch):
    csv = tmp_path / "model_comparison.csv"
    monkeypatch.setattr("src.utils.registry.COMPARISON_CSV", csv)
    log_result("test_model", "test", {"team_acc": 0.5, "team_macro_f1": 0.4,
                                       "n_eval": 10})
    df = pd.read_csv(csv)
    assert list(df.columns) == COLUMNS
    assert df.iloc[0]["model"] == "test_model"


def test_seed_row_is_present_after_seeding(tmp_path, monkeypatch):
    csv = tmp_path / "model_comparison.csv"
    monkeypatch.setattr("src.utils.registry.COMPARISON_CSV", csv)
    seed_from_baseline_2()
    df = pd.read_csv(csv)
    assert (df["model"] == "baseline_model_2").any()
```

- [ ] **Step 2: Implement `src/utils/registry.py`**

```python
"""Append-only registry of model results at reports/model_comparison.csv."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.config import BASELINE2_METRICS, REPORTS

COMPARISON_CSV = REPORTS / "model_comparison.csv"
COLUMNS = [
    "timestamp", "model", "train_set", "eval_set",
    "team_acc", "team_macro_f1", "issue_macro_f1", "issue_top3",
    "joint_exact", "north_star_coverage_95", "precision_at_95_threshold",
    "per_team_thr_coverage_95", "teams_below_floor", "ece",
    "n_eval", "git_commit",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPORTS.parent, text=True).strip()
    except Exception:  # pragma: no cover
        return ""


def log_result(model: str, eval_set: str, metrics: dict) -> None:
    row = {c: metrics.get(c) for c in COLUMNS}
    row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row["model"] = model
    row["eval_set"] = eval_set
    row["git_commit"] = _git_commit()
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = not COMPARISON_CSV.exists()
    pd.DataFrame([row]).to_csv(COMPARISON_CSV, mode="a", header=header,
                               index=False, columns=COLUMNS)


def seed_from_baseline_2() -> None:
    m = json.loads((BASELINE2_METRICS / "metrics.json").read_text())
    ns = json.loads((BASELINE2_METRICS / "north_star.json").read_text())
    op = ns["operating_points"]["at_95p"]
    log_result("baseline_model_2", "test", {
        "train_set": "3k_team",
        "team_acc": m["model"]["test"]["accuracy"],
        "team_macro_f1": m["model"]["test"]["macro_f1"],
        "north_star_coverage_95": op["coverage"],
        "precision_at_95_threshold": op["precision"],
        "teams_below_floor": ",".join(ns["guardrail"]["teams_below_floor"]),
        "ece": ns["ece_test"],
        "n_eval": ns["n_test"],
    })
```

- [ ] **Step 3: Verify tests pass**

Run: `pytest tests/test_utils_registry.py -q`
Expected: 2 passed.

- [ ] **Step 4: Seed the real registry**

Run:
```bash
F:/NLP_MSML641/.venv/Scripts/python -c "from src.utils.registry import seed_from_baseline_2; seed_from_baseline_2()"
```
Verify:
```bash
head F:/NLP_MSML641/reports/model_comparison.csv
```
Expected: header row + one `baseline_model_2` row on `test`.

- [ ] **Step 5: Commit**

```bash
cd F:/NLP_MSML641 && git add src/utils/registry.py tests/test_utils_registry.py \
    reports/model_comparison.csv
git commit -m "feat(utils): append-only results registry seeded from baseline 2"
```

---

### Task 11: `tests/test_parity_baseline2.py` — the go/no-go check

**Files:**
- Create: `tests/test_parity_baseline2.py`

- [ ] **Step 1: Write the parity test**

```python
"""Parity: our utils reproduce baseline 2's committed numbers exactly.

If this fails, STOP — every improved model comparison would be against a
different denominator than the served baseline.
"""
import json
import numpy as np
import pandas as pd
import pytest

from src.utils import config as c
from src.utils.routing import (
    coverage_curve, per_team_routing, routed_stats,
)


def _load_preds():
    df = pd.read_csv(c.BASELINE2_METRICS / "predictions_test.csv",
                     dtype={c.ID_COL: "string"})
    return df


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
    df = _load_preds()
    ns = json.loads((c.BASELINE2_METRICS / "north_star.json").read_text())
    thr = ns["operating_points"]["at_95p"]["threshold"]
    names = pd.read_csv(c.TEAM_MAPPING)
    names_lookup = dict(zip(names["team_id"], names["team_name"]))
    ours = per_team_routing(
        df["actual_team_id"].to_numpy(),
        df["predicted_team_id"].to_numpy(),
        df["confidence"].to_numpy(),
        thr, names_lookup)
    committed = pd.read_csv(c.BASELINE2_METRICS / "per_team_routing.csv")
    for col in ("team_id", "n_routed_to", "precision_routed", "auto_routed_share"):
        pd.testing.assert_series_equal(
            ours.sort_values("team_id").reset_index(drop=True)[col],
            committed.sort_values("team_id").reset_index(drop=True)[col],
            check_dtype=False, atol=1e-4, rtol=0)


@pytest.mark.slow
def test_baseline2_model_scored_on_rebuilt_test_matches_macro_f1():
    """Load the committed model.joblib and score it against our rebuilt test
    split. Confirms that our splits + preprocessing == baseline 2's."""
    import joblib
    from sklearn.metrics import f1_score
    model = joblib.load(c.BASELINE2_METRICS / "model.joblib")
    test = pd.read_parquet(c.SPLITS_V2 / "test.parquet")
    y = test[c.TEAM_COL].to_numpy()
    pred = model.predict(test[c.TEXT_COL].astype(str))
    macro = f1_score(y, pred, average="macro")
    assert abs(macro - 0.6475) < 0.001
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_parity_baseline2.py -q -m slow`
Expected: 3 passed. **If any test fails, STOP and report to user before
proceeding to plan 02.**

- [ ] **Step 3: Commit**

```bash
cd F:/NLP_MSML641 && git add tests/test_parity_baseline2.py
git commit -m "test: parity with baseline_model_2 on splits + routing harness"
```

---

## Self-review checklist

- [ ] Every §1.x subsection of the source spec has a task:
  1.1 config → Task 2; 1.2 labels → Task 4; 1.3 build_splits → Task 6;
  1.4 data → Task 7; 1.5 text → Task 3; 1.6 metrics+routing → Tasks 8+9;
  1.7 registry → Task 10; §0.1 device → Task 5.
- [ ] `src/utils/serving.py` is NOT created here (belongs to plan 04).
- [ ] All new tests marked `slow` only when they need the 1.9 GB CSV or full
  Parquets; unit tests run against synthetic data.
- [ ] Nothing modifies baseline 2 code or its committed artifacts.
- [ ] `pytest.ini` slow marker added (Task 6).
- [ ] Registry seeded with baseline 2 row on `test` (Task 10).
- [ ] Parity go/no-go test present (Task 11).

## Handoff

Next plan: `02-improved-model-1-hierarchical-svm.md`.
