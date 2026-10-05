# Improved Model 1: Hierarchical TF-IDF + LinearSVC

## Objective

Improved Model 1 routes a consumer complaint to the right support team and,
within that team, suggests the specific issue the consumer is reporting. It
uses only the written complaint narrative, with no product, company, or
metadata inputs.

| Role | Dataset column | Description |
|---|---|---|
| Input feature | `complaint_text` | Free-text complaint narrative |
| Primary target | `team_id` | One of 11 support teams (`T01`–`T11`) |
| Secondary target | `issue_id` | One of 73 canonical issues (merged variants) |

Compared with Baseline 2 (TF-IDF + Multinomial Naive Bayes, team only), this
model:

1. Replaces Naive Bayes with a one-vs-rest **Linear SVM**, trained on a
   much larger per-issue capped training set (~259 k rows vs 32 k).
2. Adds word (1–2) **and** character (3–5) n-grams.
3. Adds a **hierarchical issue head**: one LinearSVC per team, trained only
   on that team's complaints.
4. Fits **per-team routing thresholds** on validation so each team controls
   its own precision–coverage trade-off, not just a single global threshold.

The eleven teams are unchanged from Baseline 2 and are defined in
`data/processed/team_mapping.csv`.

## Data

Improved Model 1 reuses Baseline 2's partitioning rules exactly, so metrics
can be compared on the identical test set.

| Split | Period | Records | Team mix |
|---|---|---:|---|
| Training (`train_cap_issue`) | 2024-01-01 – 2025-06-30 | 257,694 | Capped at 10,000 per **issue** |
| Validation | 2025-07-01 – 2025-09-30 | 153,502 | Every eligible record (real mix) |
| Test | 2025-10-01 – 2025-12-31 | 109,834 | Every eligible record (real mix) |
| 2026 forward-drift cohort | 2026-01 – 2026-02 | 29,798 | Every eligible record |

Leakage groups that span more than one period are dropped entirely (6,867
rows across 1,949 groups). The test complaint-ID set is **identical** to
Baseline 2's committed `predictions_test.csv` — verified by
`tests/test_parity_baseline2.py`.

The training cap is per-**issue** rather than per-team so that rare issues
get full representation. Teams dominated by one issue (e.g. T01) still
contribute heavily; the cap of 10,000 per issue lets small issues contribute
all available rows.

Full partition and parity details:
`docs/metrics/improved_model_1/split_report.json`.

## Model

### Features

```python
word = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                       min_df=3, max_df=0.95, max_features=150_000,
                       sublinear_tf=True, strip_accents="unicode")
char = TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb",
                       ngram_range=(3, 5), min_df=5, max_features=150_000,
                       sublinear_tf=True)
vectorizer = FeatureUnion([("word", word), ("char", char)])
```

`clean_text` is the canonical preprocessing from
`data/preparation/advanced_text.py`.

### Hierarchy

```
team_clf   : LinearSVC(C=0.25, class_weight="balanced")  → 11 teams
issue_clfs : {team_id: LinearSVC(C=0.1, class_weight="balanced")}
```

One issue head per team, trained only on that team's training rows. Teams
with a single issue in training get a constant predictor.

### Calibration

The fitted LinearSVC team head is wrapped with
`CalibratedClassifierCV(FrozenEstimator(team_clf), method="isotonic")` and
calibrated on the **full validation set** (153,502 real-mix rows). Each
per-team issue head is calibrated the same way on that team's validation
subset, with sigmoid or softmax fallback when there are fewer than 200 rows.

### Routing

Two thresholds, both fit on validation and frozen for test:

- **Global**: lowest confidence at which overall validation precision ≥ 95%
  (`0.818`).
- **Per-team**: for each predicted team, the lowest threshold at which
  routed-to-team validation precision ≥ 95% (`T01 → 0.714`, `T05 → 0.968`,
  etc.). Teams that cannot reach 95% are never auto-routed; T02 and T11
  fall into that category.

### Hyperparameter search

Grid search on validation macro-F1. Best (`C_team=0.25`, `C_issue=0.1`)
chosen in two stages:

| C_team | val team macro-F1 |
|---:|---:|
| 0.1 | 0.6730 |
| **0.25** | **0.6742** |
| 0.5 | 0.6710 |
| 1.0 | 0.6657 |

| C_issue (oracle team) | val issue macro-F1 |
|---:|---:|
| **0.1** | **0.5158** |
| 0.25 | 0.5112 |
| 0.5 | 0.5024 |
| 1.0 | 0.4914 |

Full grid: `docs/metrics/improved_model_1/tuning.json`.

## Evaluation

### Headline (on the identical test set as Baseline 2)

| Metric | Dummy | Baseline 2 | **Improved Model 1** | Δ vs Baseline 2 |
|---|---:|---:|---:|---:|
| Accuracy | 0.147 | 0.800 | **0.844** | +0.044 |
| Macro-F1 | 0.023 | 0.647 | **0.726** | +0.079 |
| Weighted-F1 | 0.038 | 0.793 | **0.840** | +0.047 |
| ECE | — | 0.026 | **0.026** | ±0 |

### Per team (test)

| Team | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| T01 Credit Reporting | 0.884 | 0.932 | 0.907 | 57,889 |
| T02 Debt Collection | 0.809 | 0.713 | 0.758 | 16,116 |
| T03 Credit Card | 0.822 | 0.716 | 0.766 | 9,725 |
| T04 Banking Accounts | 0.773 | 0.829 | 0.800 | 10,164 |
| T05 Money Transfers | 0.741 | 0.725 | 0.733 | 5,236 |
| T06 Mortgage | 0.868 | 0.927 | 0.896 | 3,367 |
| T07 Student Loan | 0.813 | 0.766 | 0.789 | 1,517 |
| T08 Vehicle Finance | 0.765 | 0.771 | 0.768 | 2,857 |
| T09 Personal & Short-Term Loans | 0.678 | 0.585 | 0.628 | 1,640 |
| T10 Prepaid Card | 0.755 | 0.539 | 0.628 | 856 |
| T11 Debt & Credit Management | 0.748 | 0.197 | 0.312 | 467 |
| **Macro average** | | | **0.726** | — |
| **Weighted average** | | | **0.840** | — |

Full report: `docs/metrics/improved_model_1/classification_report.json`.
Confusion matrix: `docs/metrics/improved_model_1/confusion_matrix.csv`.

### North star — autonomous routing

Thresholds are fit on validation and frozen for test.

| Target | Threshold | Val coverage | **Test coverage** | **Test precision** | Routed |
|---:|---:|---:|---:|---:|---:|
| ≥ 95% precision | 0.818 | 63.9% | **61.5%** (95% CI 61.2–61.8%) | **94.8%** (95% CI 94.6–95.0%) | 67,550 |
| ≥ 90% precision | 0.604 | 86.9% | 85.3% | 89.8% | 93,707 |
| ≥ 85% precision | 0.000 | 100% | 100% | 84.4% | 109,834 |

**Baseline 2 at the same operating point: 46.5% coverage at 94.4% precision.**
Improved Model 1 lifts auto-routing by **+15 percentage points of coverage**
with marginally higher precision.

### Guardrail

Baseline 2 breached the 0.85 precision floor on T09.
Improved Model 1 keeps every team above the floor:

| Team | Routed at global threshold | Precision | Meets 0.85 floor? |
|---|---:|---:|---|
| T01 | 44,808 | 0.962 | ✓ |
| T02 | 6,862 | 0.895 | ✓ |
| T03 | 4,350 | 0.946 | ✓ |
| T04 | 4,913 | 0.922 | ✓ |
| T05 | 2,017 | 0.888 | ✓ |
| T06 | 2,259 | 0.962 | ✓ |
| T07 | 696 | 0.943 | ✓ |
| T08 | 1,101 | 0.907 | ✓ |
| T09 | 293 | 0.894 | ✓ |
| T10 | 231 | 0.952 | ✓ |
| T11 | 20 | 0.900 | ✓ |

### Per-team thresholds

New in Improved Model 1. Each team's threshold is chosen on validation to
reach ≥ 95% precision on complaints routed **to that team**. Teams that
cannot reach the target are never auto-routed.

| Team | Validation-fit threshold | Test routed | Test precision |
|---|---:|---:|---:|
| T01 | 0.714 | 51,200 | 0.943 |
| T02 | — (unreachable) | 0 | — |
| T03 | 0.876 | 3,321 | 0.963 |
| T04 | 0.890 | 3,039 | 0.946 |
| T05 | 0.968 | 59 | 0.932 |
| T06 | 0.832 | 2,153 | 0.964 |
| T07 | 0.946 | 232 | 0.991 |
| T08 | 0.956 | 66 | 0.955 |
| T09 | 0.884 | 123 | 0.951 |
| T10 | 0.888 | 151 | 0.954 |
| T11 | — (unreachable) | 0 | — |

**Overall per-team-threshold coverage: 54.9% at 94.5% precision (95% CI
94.4–94.7%).** Lower raw coverage than the global threshold, but every
auto-routed team is above 95% individually, which is a stronger guarantee for
downstream workflows.

### Issue classification (hierarchical)

Issue is predicted using the issue head of the predicted team.

| Metric | Value |
|---|---:|
| Accuracy (predicted team) | 0.561 |
| Macro-F1 (predicted team) | **0.367** |
| Accuracy (**oracle** team) | 0.652 |
| Macro-F1 (oracle team) | **0.511** |

The oracle number is the ceiling the hierarchy could reach with a perfect
team classifier — a 0.144 F1 gap to close before issue prediction stops
being bottlenecked by team errors.

**Flat ablation** (single LinearSVC over all 73 issues, team derived from
the mapping): issue macro-F1 = 0.312, team macro-F1 = 0.631.
Hierarchical > flat by **+0.055 issue F1** and **+0.095 team F1** —
confirming the hierarchy is net positive.

**Joint exact match** (team **and** issue both correct): **56.1%** of test
complaints.

Full report: `docs/metrics/improved_model_1/classification_report_issue.json`.

### Hard pair recall

Baseline 2 regularly confuses adjacent product teams. Improved Model 1
reduces the confusion but does not eliminate it:

| Pair | Metric | Value |
|---|---|---:|
| T02 Debt Collection | recall | 71.3% |
| | share predicted as T01 | 23.4% |
| T05 Money Transfers | recall | 72.5% |
| | share predicted as T04 (Banking) | 21.2% |
| T11 Debt & Credit Management | recall | 19.7% |
| | share predicted as T01 | 33.8% |
| | share predicted as T02 | 24.8% |

T11 remains the weakest team by a large margin — it has only 467 test rows
and the issue overlap with T01 and T02 is semantic (both deal with credit
disputes). This is a known limitation.

### 2026 forward-drift cohort (secondary)

| Metric | Baseline 2 | **Improved Model 1** | Δ |
|---|---:|---:|---:|
| Team macro-F1 | 0.606 | **0.697** | +0.091 |
| Team accuracy | — | 0.767 | — |
| Issue macro-F1 (predicted team) | — | 0.354 | — |
| Joint exact | — | 0.510 | — |

The 2026 cohort has already been used for evaluation and is not an untouched
holdout. It is reported for forward-drift monitoring only.

### Model comparison summary

See `reports/model_comparison.csv` for the full registry. Snapshot:

| Model | Train set | Eval | Team macro-F1 |
|---|---|---|---:|
| Dummy (most-frequent) | train_cap_issue | test | 0.023 |
| Baseline 2 (TF-IDF + NB, calibrated) | 3k_team | test | 0.647 |
| Flat issue ablation (LinearSVC on issue) | train_cap_issue | test | 0.631 |
| NB reference (Baseline 2 recipe on cap_issue) | train_cap_issue | test | 0.668 |
| **Improved Model 1 (hierarchical SVM)** | train_cap_issue | test | **0.726** |
| Baseline 2 | 3k_team | holdout_2026 | 0.606 |
| **Improved Model 1** | train_cap_issue | holdout_2026 | **0.697** |

The `nb_reference_cap_issue` row isolates the effect of the classifier from
the effect of more data: NB on 257k rows reaches 0.668 (+0.021 over
Baseline 2's 32k-row recipe), and swapping NB for the hierarchical SVM adds
another +0.058 on top of that.

## Known limitations

1. **T11 (Debt & Credit Management)** stays at F1 = 0.312. Its issues
   semantically overlap with T01 (credit reporting) and T02 (debt
   collection), and with only 467 test rows the model has little signal to
   separate them. Per-team routing never auto-routes T11.
2. **T02 (Debt Collection)** cannot reach 95% precision on its own at any
   threshold on validation, so per-team routing abstains for T02 entirely.
   The global threshold still routes T02 complaints at 89.5% precision —
   above the 85% floor but below the 95% target.
3. **Issue head is bottlenecked by the team head.** The oracle vs predicted
   gap (0.511 vs 0.367 issue macro-F1) says 0.144 F1 of issue performance
   is lost to team errors. A model that predicts team and issue jointly
   (planned for Improved Model 4) could close this gap.
4. **Linear model, no sub-word representation.** Typos and brand names
   outside the training vocabulary degrade accuracy. Character n-grams
   partially mitigate but not fully. FastText / sentence embeddings
   (Improved Model 2) are the next step.
5. **Validation protocol.** This run uses the "shared" protocol
   (calibration and thresholds both fit on the full validation set),
   identical to Baseline 2. A group-aware 50/50 "split" protocol would be
   stricter but is not run in this report.

## Reproducing

### Prerequisites

- Branch `improved_model_1`.
- `.venv` with scikit-learn 1.9.1, pyarrow, joblib (per `requirements.txt`)
  and `pytest` (per `requirements-train.txt`).
- Full source CSV at
  `data/processed/complaints_product_issues_2024_2025.csv` (gitignored).
- Committed Baseline 2 artifacts at `docs/metrics/baseline_model_2/`.

### Build the splits once

```powershell
cd F:\NLP_MSML641
.\.venv\Scripts\Activate.ps1
python -m src.utils.build_splits
```

Writes 5 Parquets to `data/cache/splits_v2/` (gitignored) and the parity
check in `split_report.json`. Takes ~5 min on first run (streams the
1.9 GB CSV to Parquet); near-instant afterwards.

### Verify parity with Baseline 2

```powershell
python -m pytest tests\test_parity_baseline2.py -q -m slow
```

Three assertions must pass before any new model is trained.

### Train

```powershell
python -m src.improved_model_1.train
```

Writes everything in §Evaluation to
`docs/metrics/improved_model_1/`. End-to-end wall clock on an RTX 4050
laptop (6 GB VRAM, 16 GB RAM): ~4.5 h with `use_char=True`, closer to
30-40 min with `--no-char` (expect ~0.5-1 team macro-F1 point lower in
exchange).

### Predict

```powershell
python -m src.improved_model_1.predict "A debt collector keeps calling me about a loan I already paid off two years ago"
```

Output JSON contains `predicted_team_id`, `predicted_team_name`,
`confidence`, `top_k` (teams), `issue_top_k` (issue IDs + labels + scores),
`route` (`"auto"` or `"review"` depending on the team's threshold), and
`model_name`.

### Full test suite

```powershell
python -m pytest -q
```

Expected: all tests pass (37 at time of writing, including the predict
contract test which runs once `model.joblib` exists).

## Generated artifacts

`docs/metrics/improved_model_1/`

| File | Purpose |
|---|---|
| `metrics.json` | Dummy vs model, hard-pair recalls, row counts |
| `classification_report.json` | Per-team precision / recall / F1 |
| `classification_report_issue.json` | Per-issue precision / recall / F1 |
| `confusion_matrix.csv` | 11×11 team confusion matrix |
| `north_star.json` | Routing coverage + precision at 95/90/85% thresholds, bootstrap CIs, ECE, guardrail |
| `per_team_routing.csv` | Per-team routing counts at the global 95% threshold |
| `per_team_thresholds.csv` | Per-team custom thresholds + test precision |
| `per_team_thresholds_summary.json` | Overall coverage + precision under per-team thresholds |
| `coverage_curve.csv` | Threshold → coverage/precision curve for plotting |
| `predictions_test.csv` | All 109,834 test predictions with issue top-3 and routing flags |
| `issue_metrics.json` | Issue accuracy / macro-F1 under predicted vs oracle team |
| `tuning.json` | Grid search log |
| `run_info.json` | Best C values, row counts, wall-clock, GPU info, artifact size, versions |
| `model.joblib` | Vectorizer + HierarchicalSVM + calibrators + thresholds (66.2 MB) |
| `issue_canonical.csv` | 73 canonical issues with team + human-readable label |
| `split_report.json` | Split counts, dropped crossing groups, parity verdict |
| `train.log` | Full training stdout |

`reports/model_comparison.csv` gains rows for improved_model_1 on test and
holdout_2026, plus flat_issue_ablation, nb_reference_cap_issue,
dummy_most_frequent and baseline_model_2 on holdout_2026.

## Related

- `docs/baseline_model_2.md` — the model this one replaces.
- `docs/implementation_plan_improved_models.md` — overall roadmap.
- `docs/plans/2026-09-30-improved-models/02-improved-model-1-hierarchical-svm.md`
  — the task-level plan this document closes out.
- `data/processed/README.md` — construction rules of the source dataset.
