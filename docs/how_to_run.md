# How to Run — Testing & Running the Models

Quick reference for teammates. All commands assume **Windows PowerShell** and
that you have the repo cloned to `F:\NLP_MSML641` (adjust the path if yours
is different).

Full write-ups live in [`improved_model_1.md`](improved_model_1.md) and
[`improved_model_1_quickstart.md`](improved_model_1_quickstart.md). This file
is the short cheat-sheet.

---

## 0. One-time setup (per machine)

```powershell
# Clone and enter
cd F:\
git clone https://github.com/farcypeer46/ClariCase.git NLP_MSML641
cd F:\NLP_MSML641

# Check out the branch with the improved model
git checkout improved_model_1

# Create virtual env (Python 3.14 recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install runtime dependencies
python -m pip install -r requirements.txt

# Install training/test dependencies (adds pytest, torch, etc.)
python -m pip install -r requirements-train.txt
```

If `.\.venv\Scripts\Activate.ps1` is blocked by execution policy, run **once**:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Verify the environment

```powershell
python -c "import sklearn, pandas, numpy, joblib; print('sklearn', sklearn.__version__)"
# Expected: sklearn 1.9.1
```

---

## 1. Every new terminal session

```powershell
cd F:\NLP_MSML641
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`.

---

## 2. Test a prediction on the improved model

### Single complaint
```powershell
python -m src.improved_model_1.predict "A debt collector keeps calling me about a loan I already paid"
```

### Several complaints
```powershell
$cases = @(
  "My credit report shows an account I never opened",
  "I sent money via Zelle and it never reached the recipient",
  "The bank charged me an overdraft fee I don't understand",
  "The dealer sold me a car loan with hidden fees"
)
foreach ($c in $cases) {
  Write-Host "`n>>> $c" -ForegroundColor Yellow
  python -m src.improved_model_1.predict $c
}
```

### Python REPL (fastest for many predictions — model loads once)
```powershell
python
```
Then inside Python:
```python
from src.improved_model_1.predict import predict, load_model
model = load_model()
predict("The bank keeps charging overdraft fees", model=model)
```
Exit with `exit()`.

### Pick a real complaint from the test set (has ground-truth labels)
```powershell
python -c "import pandas as pd; df = pd.read_parquet('data/cache/splits_v2/test.parquet').sample(3, random_state=42); print(df[['team_id','issue_id','complaint_text']].to_string())"
```
Grab a `complaint_text`, pass it to `predict`, compare the output against
the printed `team_id` / `issue_id`.

### What the output means

```json
{
  "predicted_team_id": "T02",
  "predicted_team_name": "Debt Collection Team",
  "confidence": 0.922,
  "top_k": [ ... top 3 team candidates ... ],
  "issue_top_k": [ ... top 3 issues within predicted team ... ],
  "route": "auto",
  "model_name": "improved_model_1"
}
```

- `confidence` is a calibrated probability (not a raw score).
- `route: "auto"` means confidence ≥ the predicted team's threshold
  (T01 ≥ 0.714, T05 ≥ 0.968, etc.) — safe to auto-forward.
- `route: "review"` means a human should see it first.
- T02 and T11 are **never** auto-routed (can't reach 95% precision reliably).

---

## 3. Run the automated test suite

### Everything (37 tests, ~2 minutes)
```powershell
python -m pytest -q
```

### Fast tests only (unit tests, ~5 seconds)
```powershell
python -m pytest -q -m "not slow"
```

### Only the parity check with Baseline 2 (data-heavy; must pass)
```powershell
python -m pytest tests\test_parity_baseline2.py -q -m slow
```

### Only the predict contract test
```powershell
python -m pytest tests\test_improved_model_1_predict.py -v
```

### Only a specific test file
```powershell
python -m pytest tests\test_utils_metrics.py -v
```

**Expected:** every command above ends with "N passed" and exit code 0.

---

## 4. Where to find the results

### Headline numbers (test set, 109,834 complaints)

Metric file is `F:\NLP_MSML641\docs\metrics\improved_model_1\metrics.json`:
```powershell
Get-Content F:\NLP_MSML641\docs\metrics\improved_model_1\metrics.json
```

Model vs Baseline 2:

| | Baseline 2 | Improved Model 1 |
|---|---:|---:|
| Accuracy | 0.800 | **0.844** |
| Macro F1 | 0.647 | **0.726** |
| Auto-routed @ 95% precision | 46.5% | **61.5%** |

### Side-by-side model comparison
```powershell
Import-Csv F:\NLP_MSML641\reports\model_comparison.csv | Format-Table -AutoSize
```

### Per-team details
```powershell
Import-Csv F:\NLP_MSML641\docs\metrics\improved_model_1\per_team_routing.csv | Format-Table -AutoSize
Import-Csv F:\NLP_MSML641\docs\metrics\improved_model_1\per_team_thresholds.csv | Format-Table -AutoSize
```

### Confusion matrix (open in Excel)
```powershell
Invoke-Item F:\NLP_MSML641\docs\metrics\improved_model_1\confusion_matrix.csv
```

### Every result file at a glance
```powershell
Get-ChildItem F:\NLP_MSML641\docs\metrics\improved_model_1\ | Format-Table Name, Length, LastWriteTime
```

---

## 5. Retrain the model from scratch (optional)

**Warning:** takes 30 min – 4.5 h depending on hardware and whether you use
character n-grams.

### 5a. Build the data splits (only needed once, or if source CSV changes)
```powershell
python -m src.utils.build_splits
```
Produces `data/cache/splits_v2/*.parquet` (gitignored, ~1 GB).

Takes ~5 min first time (streams the 1.9 GB CSV to Parquet), near-instant
afterwards.

### 5b. Fast training (recommended on 16 GB RAM laptops) — ~30-40 min
```powershell
python -m src.improved_model_1.train --no-char
```
Drops character n-grams. Expect ~0.5–1 team macro-F1 point lower than full.

### 5c. Full training (word + char features) — ~4.5 h on an RTX 4050 laptop
```powershell
python -m src.improved_model_1.train
```
Produces the exact numbers currently in `docs/metrics/improved_model_1/`.

### 5d. Skip reference-model comparisons (shaves ~15-20 min off either option)
```powershell
python -m src.improved_model_1.train --no-char --skip-refs
```

### 5e. Monitor a long training run (second PowerShell window)
```powershell
Get-Content -Wait F:\NLP_MSML641\docs\metrics\improved_model_1\train.log -Tail 50
```

---

## 6. Run the deployed Baseline 2 (what the Streamlit app serves)

```powershell
python -m src.baseline_model_2.predict "My credit report shows an account that is not mine"
```
Note the output here has fewer fields than Improved Model 1 — no `issue_top_k`,
no `route`, no `model_name`.

### Launch the local Streamlit app (uses Baseline 2 by default)
```powershell
python -m streamlit run streamlit_app.py
```
Open the printed `http://localhost:8501/` URL in a browser.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: src` | Make sure you're in `F:\NLP_MSML641`, not a subdirectory. |
| `cannot import name '…' from 'src.utils'` | Did you forget to install `requirements-train.txt`? Or you're on the wrong branch — do `git checkout improved_model_1`. |
| `No such file: data/processed/complaints_product_issues_2024_2025.csv` | The 1.9 GB CSV is gitignored. Pull it from the archive zip in `data/`, or ask a teammate for it. |
| `No such file: data/cache/splits_v2/*.parquet` | Run `python -m src.utils.build_splits` first (one-time, ~5 min). |
| `No such file: model.joblib` | Either run `python -m src.improved_model_1.train` first, or pull the branch where it's committed. |
| Memory climbs past 90% during training | Use `--no-char`, or edit `WORD_MAX_FEATURES = 50_000` in `src/improved_model_1/config.py`. |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Run once: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`. |
| Tests mark `slow` tests as skipped | Run with `-m slow` to include them: `python -m pytest -q -m slow`. |

---

## 8. Clean up / reset

### Remove local caches (gitignored, safe to delete)
```powershell
Remove-Item -Recurse -Force F:\NLP_MSML641\data\cache
```

### Deactivate the virtual env
```powershell
deactivate
```

### Fresh clone from scratch (nuclear)
```powershell
Remove-Item -Recurse -Force F:\NLP_MSML641
# Then re-run Section 0 setup
```
