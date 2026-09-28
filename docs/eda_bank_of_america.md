# Bank of America complaint EDA

Cleaning and exploratory analysis of 16,407 complaints across eight product categories. Complaint dates range from January 1, 2024 to February 2, 2026.

## Files

- `original_dataset.csv`: unchanged copy of the prepared dataset used by the notebook.
- `consumer_complaint_eda.ipynb`: executed notebook with data quality checks, text-length analysis, product distributions, and final validation.
- `cleaned_consumer_complaints.csv`: notebook output, retaining all 16,407 rows and seven columns.

The notebook normalizes repeated and leading or trailing whitespace in 10,190 narratives. Wording, capitalization, punctuation, numbers, and redactions are preserved. There are no models, embeddings, or train/test splits.

## Run

Use Python 3.12. From the repository root:

```bash
cd consumer_complaint_eda
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab consumer_complaint_eda.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Run the notebook from beginning to end. It reads `original_dataset.csv` and writes `cleaned_consumer_complaints.csv` in the same folder. The original CSV must be present to rerun the analysis.

## Dataset scope

The source is the [official CFPB narrative archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/). The input here is a prepared subset, not the full Bank of America export.

Before this notebook, preparation retained products with at least 100 original narratives, removed duplicate normalized text and groups with conflicting product labels, and excluded text matching the supplied JPMorgan dataset. The eight retained categories are checking or savings accounts, credit cards, credit reporting, money transfers or services, debt collection, prepaid cards, mortgages, and vehicle loans or leases. Original CFPB label strings are retained in the CSV.

The final subset has no matching normalized narratives against the supplied 22,465-row JPMorgan file or the 452 Morgan Stanley narratives found across all 21 official archive files. Matching collapsed whitespace and ignored case. The teammate's exact Morgan Stanley file was unavailable, and near-duplicate wording was not covered by this comparison.

These earlier selection steps are not repeated in this notebook. Repeat cross-company checks when combining different or updated team datasets. Product labels are consumer-selected categories, and the category proportions describe this selected dataset.
