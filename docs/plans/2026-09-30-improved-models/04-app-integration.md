# 04 — App Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Plug an improved model (default: model 1) into `streamlit_app.py`
without breaking the currently-deployed app. Baseline 2 remains the **default**
served model until the user explicitly switches. All storage columns are added
as **nullable** so existing complaint rows are untouched.

**Architecture:** A thin serving adapter (`src/utils/serving.py`) selects the
active model via env var / Streamlit secret / hard-coded default. The Streamlit
app calls `route(text, name, model)` instead of importing baseline 2 directly.
`build_record()` learns four new nullable fields; a SQL migration adds them to
Supabase; `LocalStore` self-migrates SQLite on init.

**Tech Stack:** Streamlit, importlib, sqlite3 (`PRAGMA table_info`), Supabase
Python client, pytest.

**Prerequisites:**
- Plan 02 complete and the user approved the swap.
- Fresh branch: `git checkout -b app/improved-model-1` from the current HEAD of
  `improved_model_1` (or `main` if model 1 has been merged).

---

## File layout produced

```
src/utils/
└── serving.py                                # new — model adapter
src/app/
├── storage.py                                # modified: nullable columns + local migration
└── migrations/
    └── 002_add_issue_columns.sql             # new — SQL for Supabase (run by user)
streamlit_app.py                              # modified: use route(), show issue line
tests/
└── test_app_contract.py                     # new
docs/metrics/improved_model_1/
└── model.joblib                              # committed here for the first time
```

---

### Task 1: `src/utils/serving.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_app_contract.py`:
```python
import pytest
from unittest.mock import patch

from src.utils import serving as S


def test_active_model_name_defaults_to_baseline_2(monkeypatch):
    monkeypatch.delenv("CLARICASE_MODEL", raising=False)
    with patch.object(S, "_secrets_model_name", return_value=None):
        assert S.active_model_name() == "baseline_model_2"


def test_active_model_name_respects_env(monkeypatch):
    monkeypatch.setenv("CLARICASE_MODEL", "improved_model_1")
    assert S.active_model_name() == "improved_model_1"


def test_route_returns_baseline_keys_and_fills_none_for_missing():
    class FakeModel:
        pass
    with patch.object(S, "load_router", return_value=("baseline_model_2", None)):
        with patch("src.baseline_model_2.predict.predict",
                   return_value={
                       "predicted_team_id": "T02",
                       "predicted_team_name": "Debt Collection Team",
                       "confidence": 0.9,
                       "top_k": [],
                   }):
            out = S.route("hi", "baseline_model_2", None)
    for k in ("predicted_team_id", "predicted_team_name", "confidence",
              "top_k", "issue_top_k", "route", "model_name"):
        assert k in out
    assert out["issue_top_k"] is None
    assert out["route"] is None
    assert out["model_name"] == "baseline_model_2"
```

- [ ] **Step 2: Implement `src/utils/serving.py`**

```python
"""Model adapter used by the Streamlit app."""
from __future__ import annotations

import importlib
import os
from typing import Any

MODELS = {
    "baseline_model_2": ("src.baseline_model_2.predict", "load_model", "predict"),
    "improved_model_1": ("src.improved_model_1.predict", "load_model", "predict"),
}

_BASELINE_KEYS = ("predicted_team_id", "predicted_team_name", "confidence",
                  "top_k")
_OPTIONAL_KEYS = ("issue_top_k", "route", "model_name")


def _secrets_model_name() -> str | None:
    try:
        import streamlit as st  # lazy; not a hard dep of utils
        return st.secrets["model"]["name"]
    except Exception:
        return None


def active_model_name() -> str:
    return (os.environ.get("CLARICASE_MODEL")
            or _secrets_model_name()
            or "baseline_model_2")


def load_router(name: str | None = None) -> tuple[str, Any]:
    resolved = name or active_model_name()
    if resolved not in MODELS:
        raise KeyError(f"Unknown model {resolved!r}; known: {list(MODELS)}")
    module_name, loader_attr, _ = MODELS[resolved]
    mod = importlib.import_module(module_name)
    loader = getattr(mod, loader_attr)
    return resolved, loader()


def route(text: str, name: str, model: Any, top_k: int = 3) -> dict:
    module_name, _, predict_attr = MODELS[name]
    predict_fn = getattr(importlib.import_module(module_name), predict_attr)
    try:
        raw = predict_fn(text, top_k=top_k, model=model)
    except TypeError:
        # baseline_model_1 may not accept model=...
        raw = predict_fn(text, top_k=top_k)

    out = {k: raw.get(k) for k in _BASELINE_KEYS}
    for k in _OPTIONAL_KEYS:
        out[k] = raw.get(k)
    if out["model_name"] is None:
        out["model_name"] = name
    return out
```

- [ ] **Step 3: Verify**

Run: `pytest tests/test_app_contract.py::test_active_model_name_defaults_to_baseline_2 \
    tests/test_app_contract.py::test_active_model_name_respects_env \
    tests/test_app_contract.py::test_route_returns_baseline_keys_and_fills_none_for_missing -q`

Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
cd F:/NLP_MSML641 && git add src/utils/serving.py tests/test_app_contract.py
git commit -m "feat(utils): serving adapter with model registry and env override"
```

---

### Task 2: Storage migration — Supabase SQL + local SQLite auto-migrate

- [ ] **Step 1: Write the SQL migration**

Create `src/app/migrations/002_add_issue_columns.sql`:
```sql
-- Adds nullable columns for improved-model routing metadata.
-- Run this once against the deployed Supabase project (SQL editor).
alter table public.complaints add column if not exists issue_id text;
alter table public.complaints add column if not exists issue_label text;
alter table public.complaints add column if not exists route text;
alter table public.complaints add column if not exists model_name text;
```

- [ ] **Step 2: Add a reference comment to `src/app/schema.sql`**

Read `src/app/schema.sql`; append:
```sql
-- See migrations/002_add_issue_columns.sql for the additive columns used by
-- improved models. schema.sql itself is not edited so the initial deploy
-- history stays reproducible.
```

- [ ] **Step 3: Write the failing local-migration test**

Add to `tests/test_app_contract.py`:
```python
def test_local_store_auto_migrates_old_schema(tmp_path, monkeypatch):
    """Simulate a pre-migration SQLite file, then confirm LocalStore adds columns."""
    import sqlite3
    db = tmp_path / "complaints.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE complaints (
                tracking_id TEXT PRIMARY KEY, submitted_at TEXT,
                complaint_text TEXT, team_id TEXT, team_name TEXT,
                confidence REAL, status TEXT
            );
        """)

    from src.app.storage import LocalStore
    store = LocalStore(db)  # __init__ should auto-migrate
    with sqlite3.connect(db) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(complaints)")]
    for new in ("issue_id", "issue_label", "route", "model_name"):
        assert new in cols
```

- [ ] **Step 4: Modify `src/app/storage.py`**

Read the current file, then:
- Add a `_MIGRATIONS` list of `(column, sql)` tuples for the four new columns.
- In `LocalStore.__init__` (or wherever the connection is set up), after
  creating the table, run `PRAGMA table_info(complaints)` and `ALTER TABLE
  complaints ADD COLUMN <name> <type>` for each missing entry.
- Extend `COLUMNS` (or the equivalent constant) with the four new field names.
- Modify `build_record(text, prediction)` to fill new fields via
  `prediction.get("issue_top_k", None)` (extract the top-1 label), and
  `prediction.get("route")`, `prediction.get("model_name")`. Keep baseline 2
  behaviour untouched (`.get` defaults to `None`).
- For `SupabaseStore.add`, gate the new keys behind
  `self.supports_issue_fields`, a bool set once by attempting
  `select("issue_id").limit(1)` and catching the error.

- [ ] **Step 5: Verify all tests pass**

Run: `pytest tests/test_app_contract.py -q`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
cd F:/NLP_MSML641 && git add src/app/migrations/002_add_issue_columns.sql \
    src/app/schema.sql src/app/storage.py tests/test_app_contract.py
git commit -m "feat(app): additive nullable columns for issue/route/model metadata"
```

---

### Task 3: `streamlit_app.py` edit — call `route()` instead of baseline 2 directly

- [ ] **Step 1: Read `streamlit_app.py`** to locate the current
  `from src.baseline_model_2.predict import load_model, predict` import and the
  `handle_submit` function.

- [ ] **Step 2: Edit**

Replace:
```python
from src.baseline_model_2.predict import load_model, predict
```
with:
```python
from src.utils.serving import load_router, route
```

Replace the cached `get_model()` (or equivalent) with:
```python
@st.cache_resource
def get_router():
    return load_router()  # returns (name, model)
```

Replace the prediction call site (`predict(text)` or `predict(text, model=...)`):
```python
name, model = get_router()
prediction = route(text, name, model, top_k=3)
```

Under the success message, add:
```python
issue_top = prediction.get("issue_top_k")
if issue_top:
    st.write(f"We understood this as: {issue_top[0]['issue_label']}")
```

For status:
```python
status = "Under review" if prediction.get("route") == "review" else "Received"
```

Track tab: when displaying a stored complaint, show the `issue_label` field if
present (`if row.get("issue_label"):`).

- [ ] **Step 3: Manual sanity check**

Run:
```bash
cd F:/NLP_MSML641 && F:/NLP_MSML641/.venv/Scripts/python -m streamlit run streamlit_app.py
```
Submit a complaint. Confirm:
1. Default (env not set) → uses baseline 2, works as before.
2. Set `$env:CLARICASE_MODEL="improved_model_1"`, restart, submit again → uses
   model 1, issue line appears, status may be "Under review".

If `improved_model_1/model.joblib` is not yet on disk, the app should still boot
against baseline 2 by default; test failure means the fallback is broken.

- [ ] **Step 4: Commit**

```bash
cd F:/NLP_MSML641 && git add streamlit_app.py
git commit -m "feat(app): route via serving adapter; show issue label when present"
```

---

### Task 4: Commit `improved_model_1/model.joblib` (only now)

Baseline 2's serving path is untouched, and the user has explicitly approved
the swap. The improved-model artifact can now be committed for the app to load.

- [ ] **Step 1: Verify size ≤ 95 MB**

Run:
```bash
ls -lh F:/NLP_MSML641/docs/metrics/improved_model_1/model.joblib
```
Expected: under 95 MB. If larger, STOP — go back to model 1 plan Task 5 and
either drop `char` features or lower `word_max`.

- [ ] **Step 2: Stage and commit only the model file**

```bash
cd F:/NLP_MSML641 && git add docs/metrics/improved_model_1/model.joblib
git commit -m "chore(improved_model_1): commit served model.joblib after review"
```

---

### Task 5: `supabase` in `requirements.txt` (if plan 00 flagged it)

- [ ] **Step 1: Confirm still missing**

Run:
```bash
grep -n '^supabase' F:/NLP_MSML641/requirements.txt || echo MISSING
```

- [ ] **Step 2: If MISSING**, add it and commit (the app imports it):

```bash
cd F:/NLP_MSML641 && echo "supabase" >> requirements.txt
git add requirements.txt
git commit -m "chore(deps): add supabase to runtime requirements"
```

---

### Task 6: Push branch, open PR, hand SQL migration to user

- [ ] **Step 1: Push**

```bash
cd F:/NLP_MSML641 && git push -u origin app/improved-model-1
```

- [ ] **Step 2: Print the migration path** to the user:

> "Run this once in the Supabase SQL editor:
> `src/app/migrations/002_add_issue_columns.sql`
> After it succeeds, in Streamlit Cloud secrets set `[model] name =
> "improved_model_1"` and redeploy. To roll back, delete that setting."

- [ ] **Step 3: Open the PR** with the acceptance criteria filled in.

---

## Acceptance criteria (from spec §5)

- [ ] Baseline 2 remains the default served model in absence of env var / secret.
- [ ] `route()` output contains every key `build_record()` needs, for both
  baseline 2 and model 1 (`tests/test_app_contract.py`).
- [ ] `LocalStore` migrates an old SQLite file (`tests/test_app_contract.py`).
- [ ] Supabase migration SQL is committed but **not** auto-applied — the user
  runs it.
- [ ] `model.joblib` for model 1 is ≤ 95 MB and committed only after review.
- [ ] `requirements.txt` includes `supabase` (added in plan 00 Task 4 or here).
- [ ] Manual smoke test in Task 3 Step 3 passes for both default and
  `CLARICASE_MODEL=improved_model_1`.

## Self-review

- [ ] No torch import at module load of `src/utils/serving.py`.
- [ ] No change to `src/baseline_model_2/`.
- [ ] `src/app/schema.sql` gets a reference comment only — no data DDL.
- [ ] Streamlit UI still works when the improved model returns `None` for
  `issue_top_k` / `route`.

## Handoff

Await user confirmation that the Supabase migration ran successfully before
publishing the model-swap secret. Reverting is one-line: unset
`CLARICASE_MODEL` / delete the secret and redeploy.
