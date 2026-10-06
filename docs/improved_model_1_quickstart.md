# Improved Model 1 — Quickstart & Results

Short teammate-facing guide. Full write-up is in
[`docs/improved_model_1.md`](improved_model_1.md).

Model type: **hierarchical TF-IDF + Linear SVM** (team then issue), isotonic
calibration on validation, global + per-team routing thresholds.

Status: trained on branch `improved_model_1`, and served by the Streamlit
app on this branch (live once the branch is deployed).

---

## Results at a glance

### Team classification (test set, Oct–Dec 2025, 109,834 complaints — identical to Baseline 2)

| Metric | Dummy | Baseline 2 | **Improved Model 1** | Lift vs B2 |
|---|---:|---:|---:|---:|
| Accuracy | 0.147 | 0.800 | **0.844** | +4.4 pp |
| Macro F1 | 0.023 | 0.647 | **0.726** | +7.9 pp |
| Weighted F1 | 0.038 | 0.793 | **0.840** | +4.7 pp |
| ECE (calibration error) | — | 0.026 | 0.026 | ±0 |

### Routing (how often we can auto-route vs sending to a human reviewer)

| Precision target | Threshold (val-fit) | Val coverage | **Test coverage** | **Test precision** |
|---|---:|---:|---:|---:|
| ≥ 95% | 0.818 | 63.9% | **61.5%** (CI 61.2–61.8%) | **94.8%** (CI 94.6–95.0%) |
| ≥ 90% | 0.604 | 86.9% | 85.3% | 89.8% |
| ≥ 85% | 0.000 | 100% | 100% | 84.4% |

Baseline 2 at the same ≥95% target: 46.5% test coverage at 94.4% precision.
**Improved Model 1 auto-routes +15 percentage points more complaints** at
slightly higher precision.

### Per-team guardrail (do all teams clear the 0.85 precision floor?)

**Yes — every team passes.** Baseline 2 breached T09 (0.84); Improved Model 1
pulls T09 to 0.89 and keeps every team ≥ 0.85.

| Team | Routed @ global threshold | Precision | Floor met? |
|---|---:|---:|---|
| T01 Credit Reporting | 44,808 | 0.962 | ✓ |
| T02 Debt Collection | 6,862 | 0.895 | ✓ |
| T03 Credit Card | 4,350 | 0.946 | ✓ |
| T04 Banking Accounts | 4,913 | 0.922 | ✓ |
| T05 Money Transfers | 2,017 | 0.888 | ✓ |
| T06 Mortgage | 2,259 | 0.962 | ✓ |
| T07 Student Loan | 696 | 0.943 | ✓ |
| T08 Vehicle Finance | 1,101 | 0.907 | ✓ |
| T09 Personal & Short-Term Loans | 293 | 0.894 | ✓ |
| T10 Prepaid Card | 231 | 0.952 | ✓ |
| T11 Debt & Credit Management | 20 | 0.900 | ✓ |

### Per-team F1

| Team | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| T01 Credit Reporting | 0.884 | 0.932 | **0.907** | 57,889 |
| T06 Mortgage | 0.868 | 0.927 | **0.896** | 3,367 |
| T04 Banking Accounts | 0.773 | 0.829 | **0.800** | 10,164 |
| T07 Student Loan | 0.813 | 0.766 | **0.789** | 1,517 |
| T08 Vehicle Finance | 0.765 | 0.771 | **0.768** | 2,857 |
| T03 Credit Card | 0.822 | 0.716 | **0.766** | 9,725 |
| T02 Debt Collection | 0.809 | 0.713 | **0.758** | 16,116 |
| T05 Money Transfers | 0.741 | 0.725 | **0.733** | 5,236 |
| T09 Personal Loans | 0.678 | 0.585 | **0.628** | 1,640 |
| T10 Prepaid Card | 0.755 | 0.539 | **0.628** | 856 |
| T11 Debt & Credit Mgmt | 0.748 | 0.197 | **0.312** | 467 |
| **Macro average** | | | **0.726** | — |

T11 remains the weakest team — tiny support (467 rows) and semantic overlap
with T01 and T02.

### Issue classification (new in Improved Model 1)

Baseline 2 did not predict issues. Improved Model 1 adds a hierarchical issue
head per team.

| Metric | Value |
|---|---:|
| Issue accuracy (predicted team) | 0.561 |
| Issue macro-F1 (predicted team) | **0.367** |
| Issue accuracy (**oracle** team) | 0.652 |
| Issue macro-F1 (oracle team) | **0.511** (ceiling) |
| Flat-ablation macro-F1 (single LinearSVC over all 73 issues) | 0.312 |
| **Joint exact match** (team AND issue right) | **56.1%** |

Hierarchical > flat by +0.055 F1 on issue and +0.095 F1 on team — the
hierarchy is net positive.

### Forward-drift cohort (Jan–Feb 2026, 29,798 complaints)

Secondary check. Already used for evaluation — not a fresh holdout.

| Metric | Baseline 2 | **Improved Model 1** | Lift |
|---|---:|---:|---:|
| Team macro-F1 | 0.606 | **0.697** | +9.1 pp |
| Team accuracy | — | 0.767 | — |
| Issue macro-F1 | — | 0.354 | — |

---

## How to test the model

### 0. One-time setup

```powershell
cd F:\NLP_MSML641
.\.venv\Scripts\Activate.ps1          # prompt should now show "(.venv)"
python -c "import torch, sklearn; print(torch.__version__, sklearn.__version__)"
# Expected: 2.14.0+cu126 1.9.1
```

### 1. Predict on a single complaint (CLI)

```powershell
python -m src.improved_model_1.predict "A debt collector keeps calling me about a loan I already paid off two years ago"
```

Output is JSON:
```json
{
  "predicted_team_id": "T02",
  "predicted_team_name": "Debt Collection Team",
  "confidence": 0.922,
  "top_k": [
    {"team_id": "T02", "team_name": "Debt Collection Team", "confidence": 0.922},
    {"team_id": "T01", "team_name": "Credit Reporting Team", "confidence": 0.048},
    {"team_id": "T09", "team_name": "Personal & Short-Term Loans Team", "confidence": 0.012}
  ],
  "issue_top_k": [
    {"issue_id": "T02_I002", "issue_label": "Debt collection :: Communication tactics", "confidence": 0.491},
    {"issue_id": "T02_I001", "issue_label": "Debt collection :: Attempts to collect debt not owed", "confidence": 0.420},
    {"issue_id": "T02_I006", "issue_label": "Debt collection :: Took or threatened to take negative or legal action", "confidence": 0.025}
  ],
  "route": "auto",
  "model_name": "improved_model_1"
}
```

Field cheat sheet:
- `confidence` ≥ 0.818 (global threshold) → `route: "auto"`
- `route: "review"` → send to human queue
- `top_k`: top-3 team candidates. If top-1 and top-2 are close, the model is uncertain.
- `issue_top_k`: top-3 issues **within the predicted team**, with human labels.

### 2. Predict on several complaints at once

```powershell
$cases = @(
  "A debt collector keeps calling me about a loan I already paid off two years ago",
  "My credit report shows an account I never opened",
  "The bank charged me an overdraft fee I don't understand",
  "I sent money via Zelle and it never reached the recipient",
  "The dealership sold me a car loan with fees I never agreed to",
  "My mortgage servicer isn't applying my payment correctly",
  "My student loan servicer is reporting the wrong balance"
)
foreach ($c in $cases) {
  Write-Host "`n>>> $c" -ForegroundColor Yellow
  python -m src.improved_model_1.predict $c
}
```

### 3. Interactive Python REPL (fastest for many predictions)

```powershell
python
```

```python
from src.improved_model_1.predict import predict, load_model
model = load_model()      # ~5 sec, loads the 66 MB artifact once

predict("The bank keeps charging overdraft fees even when I am not overdrawn", model=model)
predict("Someone used my debit card to buy things online", model=model)
```

Exit with `exit()` or Ctrl+Z then Enter.

### 4. Try real complaints (with ground-truth labels) from the test set

Grab 5 random real complaints:

```powershell
python -c "import pandas as pd; df = pd.read_parquet('data/cache/splits_v2/test.parquet').sample(5, random_state=1); print(df[['team_id','issue_id','complaint_text']].to_string())"
```

Copy any `complaint_text`, pass it to `predict`, compare the model output
against the printed `team_id` / `issue_id`.

### 5. Run the automated test suite

```powershell
python -m pytest -q               # 37 tests, ~1:50 (includes the 1.9 GB CSV parity check)
python -m pytest -q -m "not slow" # fast tests only, ~5 sec
python -m pytest tests\test_improved_model_1_predict.py -v   # just the predict contract test
```

Expected: 37 passed.

---

## Where the result files live

```
F:\NLP_MSML641\docs\metrics\improved_model_1\
├── model.joblib                      # 66 MB — the trained model
├── metrics.json                      # headline accuracy, F1, hard pairs
├── north_star.json                   # routing coverage/precision + bootstrap CIs
├── per_team_routing.csv              # per-team precision at the global threshold
├── per_team_thresholds.csv           # per-team custom thresholds
├── per_team_thresholds_summary.json  # overall per-team-threshold routing stats
├── coverage_curve.csv                # threshold -> (coverage, precision) curve
├── predictions_test.csv              # all 109,834 test predictions with issue top-3
├── classification_report.json        # per-team precision / recall / F1
├── classification_report_issue.json  # per-issue precision / recall / F1
├── confusion_matrix.csv              # 11x11
├── issue_metrics.json                # issue F1 under predicted vs oracle team
├── tuning.json                       # grid-search log
├── run_info.json                     # best C values, timings, versions, GPU info
└── train.log                         # full training stdout
```

Model comparison summary (apples-to-apples across all models we tried):

```
F:\NLP_MSML641\reports\model_comparison.csv
```

View in PowerShell:
```powershell
Import-Csv F:\NLP_MSML641\reports\model_comparison.csv | Format-Table -AutoSize
```

Or in Python:
```python
import pandas as pd
pd.read_csv("reports/model_comparison.csv")
```

---

## Known limitations (to set expectations)

1. **T11 (Debt & Credit Mgmt)** F1 = 0.312. Only 467 test rows, and the
   issue overlap with T01 and T02 is semantic. Per-team routing abstains
   for T11 entirely.
2. **T02 (Debt Collection)** can't reach 95% precision on validation at
   any threshold, so per-team routing abstains for T02. The global
   threshold still routes T02 at 89.5% precision (above the 85% floor).
3. **Issue head is bottlenecked by the team head.** Oracle vs predicted
   gap of 0.144 F1 — a joint team+issue model could close it.
4. **Linear model, no sub-word representation.** Typos and brand names
   outside the training vocabulary hurt accuracy. FastText / sentence
   embeddings are planned for Improved Model 2.

---

## Next steps

| Decision | Options |
|---|---|
| Commit this work? | 3-commit split (code / artifacts / docs) or 1 big commit |
| Deploy this to the Streamlit app? | Plan 04 (`docs/plans/2026-09-30-improved-models/04-app-integration.md`) |
| Try to beat it with embeddings? | Plan 03 (`docs/plans/2026-09-30-improved-models/03-improved-model-2-embeddings.md`) |
| Try transformer models? | Plan 05 roadmap (DeBERTa / ModernBERT with GPU) |
