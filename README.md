# ClariCase

We help financial services teams understand consumer complaints and identify actionable patterns more efficiently than manual review.

**Repository:** https://github.com/farcypeer46/ClariCase

---

## Team

| Name | Role |
|------|-----|
| Chaitanya Bagul | Developer, Engineering |
| Sukriti Srivastava | Developer, Product |
| Salman Farcy | Developer, Data and Evaluation |
| Sriramm S S | Developer, Users and Research |

## Local setup

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Data pipeline

Run the modules from the repository root so imports resolve correctly:

```powershell
# Load the raw CSV and display label counts
python -m src.data.ingest

# Rebuild docs/data_profile.md
python -m src.data.explore

# Clean, deduplicate, and display the temporal train/test split
python -m src.data.prepare
```

The raw input is `data/raw/complaints.csv`. Cleaning is performed in memory;
`prepare.py` does not write a processed copy to disk.

## Baseline classifier

Train and evaluate the TF-IDF + LinearSVC baseline:

```powershell
python -m src.models.baseline
```

Outputs are written to `reports/baseline/`: the fitted model, metrics,
classification report, confusion matrix, and test predictions.
