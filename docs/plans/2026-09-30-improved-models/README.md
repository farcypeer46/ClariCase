# Improved Models — Execution Plans (2026-09-30)

Sub-plans derived from `docs/implementation_plan_improved_models.md`. Each plan is
self-contained and produces working, testable software on its own.

## Reading order

| # | File | Section of source spec | Depends on |
|---|---|---|---|
| 00 | `00-environment-and-repo-hygiene.md` | §0 (ground rules, GPU, cleanup) | — |
| 01 | `01-shared-foundation-utils.md` | §1 (`src/utils/`) + parity test | 00 |
| 02 | `02-improved-model-1-hierarchical-svm.md` | §2 (TF-IDF + LinearSVC hierarchy) | 01 |
| 03 | `03-improved-model-2-embeddings.md` | §3 (word / sentence embedding variants) | 01, 02 |
| 04 | `04-app-integration.md` | §5 (serving adapter, storage migration) | 02 (or later) |
| 05 | `05-transformer-roadmap.md` | §4 (models 3/4/5 — stub only) | 01, 02 |

## Ground rules that every sub-plan inherits

1. **Do not modify** `src/baseline_model_1/`, `src/baseline_model_2/`, `src/data/`,
   `data/preparation/`, `data/processed/splits/`, or
   `docs/metrics/baseline_model_[12]/`. **Never run
   `python -m src.baseline_model_2.train`** — it would overwrite the served
   artifact.
2. Work on a branch per improvement (`improved_model_1`, `improved_model_2`, …).
   Never `git add -A` / `git add .`. Stage files by explicit path.
3. Everything reproducible: `RANDOM_SEED = 42`. Log `sklearn.__version__`,
   `torch.__version__`, `gpu_info()` to each run's `run_info.json`.
4. Model input is `complaint_text` only. Never feed `product`, `sub_product`,
   `issue`, `sub_issue`, `team_*`, `product_issue_*`, IDs, group hashes,
   `company`, or `source_*` columns to a model.
5. Vectorizers / embedding models / class weights / scalers fit on **train**.
   Calibration and thresholds fit on **validation**. Test and 2026 cohort are
   scored only.
6. `requirements.txt` is what Streamlit Community Cloud installs — keep it
   runtime-minimal. Training-only libs go in `requirements-train.txt`.
7. **Do not upgrade scikit-learn** (currently 1.9.1; baseline 2's `model.joblib`
   was saved with it).

## Execution modes (per writing-plans skill)

Each plan has task-level checkboxes (`- [ ]`). Two execution options:

- **Subagent-Driven** — dispatch a fresh subagent per task, review between tasks.
- **Inline Execution** — execute tasks in-session using
  `superpowers:executing-plans`.

Ask the user which mode before starting any plan.
