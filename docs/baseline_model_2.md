# Baseline Model 1 — TF-IDF + Multinomial Naive Bayes (Team Classifier)

## Objective

Route a consumer complaint to one of the **11 support teams** using only the
written complaint narrative. The prediction target is `team_id` (T01–T11) as
defined in `data/processed/team_mapping.csv`.

| Role | Dataset column | Description |
|---|---|---|
| Input feature | `complaint_text` | Free-text complaint narrative |
| Target | `team_id` | One of 11 team IDs from `team_mapping.csv` |
| Group key | `leakage_group_id` | Enforces group-aware splitting |

## Team labels

| team_id | team_name |
|---|---|
| T01 | Credit Reporting Team |
| T02 | Debt Collection Team |
| T03 | Credit Card Team |
| T04 | Banking Accounts Team |
| T05 | Money Transfers & Digital Currency Team |
| T06 | Mortgage Team |
| T07 | Student Loan Team |
| T08 | Vehicle Finance Team |
| T09 | Personal & Short-Term Loans Team |
| T10 | Prepaid Card Team |
| T11 | Debt & Credit Management Team |

## Directory layout

```
NLP_MSML641/
├── data/
│   └── processed/
│       ├── complaints_product_issues_2024_2025.csv   (1.9 GB, out-of-band)
│       ├── team_mapping.csv                          (tracked)
│       ├── issue_mapping.csv                         (tracked)
│       └── splits/                                   (generated, ignored)
│           ├── train.csv
│           ├── validation.csv
│           └── test.csv
└── src/
    └── baseline_model_1/
        ├── config.py         # paths, seed, split fractions
        ├── data_loader.py    # stratified sample + group-aware split
        ├── train.py          # TF-IDF + MultinomialNB pipeline
        ├── predict.py        # single-complaint inference
        └── artifacts/        # generated
            ├── model.joblib              (ignored)
            ├── metrics.json              (tracked)
            ├── classification_report.json(tracked)
            ├── confusion_matrix.csv      (tracked)
            └── predictions_test.csv      (tracked)
```

## Prerequisites

1. Python 3.10+
2. The 1.9 GB processed CSV — **not in git** — must be placed at:
   ```
   data/processed/complaints_product_issues_2024_2025.csv
   ```
   Ask a teammate for the file (shared via Drive / S3 / OneDrive).

## Setup

```bash
git clone <repo-url> NLP_MSML641
cd NLP_MSML641
git checkout baseline_model_2

python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# bash / Git Bash
source .venv/Scripts/activate

pip install -r requirements.txt
```

## Train and evaluate

Single command — builds splits if missing, fits the model, writes all
artifacts:

```bash
python -m src.baseline_model_1.train
```

Expected duration on a typical laptop:
- First run: ~3–5 min (dominated by the CSV scan for stratified sampling)
- Subsequent runs: ~30 s (splits cached under `data/processed/splits/`)

### Configuration knobs

Edit `src/baseline_model_1/config.py`:

| Constant | Default | Purpose |
|---|---|---|
| `SAMPLE_PER_TEAM` | 3000 | Max records per team drawn from the full CSV |
| `CHUNKSIZE` | 100_000 | Rows read per pandas chunk |
| `RANDOM_SEED` | 42 | Deterministic sample + split |
| `TRAIN_FRAC` | 0.60 | Train share |
| `VAL_FRAC` | 0.20 | Validation share |
| `TEST_FRAC` | 0.20 | Test share |

Or override via CLI flags on `train.py`:

```bash
python -m src.baseline_model_1.train --max-features 100000 --output-dir custom/dir
python -m src.baseline_model_1.train --quiet
```

## Predict on a new complaint

```bash
python -m src.baseline_model_1.predict "I was charged twice on my credit card and the bank refuses to refund"
```

Returns JSON with predicted `team_id`, `team_name`, confidence, and top-3
alternatives. Requires artifacts from a previous training run.

## Regenerate from scratch

```bash
rm -rf data/processed/splits/
rm -rf src/baseline_model_1/artifacts/
python -m src.baseline_model_1.train
```

## Model design

- **Vectorizer**: `TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.98, max_features=75_000, sublinear_tf=True, strip_accents='unicode')`
- **Classifier**: `MultinomialNB(alpha=0.1, fit_prior=False)`
- **Class balance**: sample weights computed from
  `sklearn.utils.class_weight.compute_sample_weight('balanced', y_train)`
  and passed to `MultinomialNB.fit` — compensates for the 62% Credit Reporting
  dominance without changing the class prior.
- **Split policy**: `GroupShuffleSplit` on `leakage_group_id`, applied twice
  (test carve-out, then val carve-out) so template near-duplicates never
  straddle partitions.

## Evaluation

Written to `src/baseline_model_1/artifacts/`:

| File | Contents |
|---|---|
| `metrics.json` | Accuracy / macro-F1 / weighted-F1 on val + test (dummy baseline included) |
| `classification_report.json` | Per-team precision, recall, F1, support |
| `confusion_matrix.csv` | 11×11 test confusion matrix, actual × predicted |
| `predictions_test.csv` | Row-level test predictions with confidence |
| `model.joblib` | Serialized sklearn Pipeline (ignored by git) |

## Reference baseline

The `DummyClassifier(strategy='most_frequent')` baseline is included in
`metrics.json` so improvements over "always predict the majority team" are
verifiable at a glance.

## Common gotchas

- **`git add data/` will fail the push** — the 1.9 GB CSV exceeds GitHub's
  100 MB per-file hard limit. `.gitignore` protects you, but only on this
  branch.
- **`model.joblib` differences across sklearn versions** — metrics should
  match within ~0.01 F1 given the same seed, but the serialized blob may
  differ byte-for-byte.
- **`ConvergenceWarning`** from sklearn is safe to ignore for this baseline;
  Naive Bayes has a closed-form solution and does not iterate.

## Related documents

- `docs/baseline_model.md` — earlier product-classification baseline (different
  target: `Product` field, not team)
- `data/processed/README.md` — dataset construction rules and reconciliation
