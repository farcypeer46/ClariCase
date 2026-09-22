# Wells Fargo complaint EDA

Cleaning and exploratory analysis of 16,883 complaints across eight product categories. Complaint dates range from January 1, 2024 to February 2, 2026.

## Files

- `original_dataset.csv`: unchanged copy of the prepared dataset used by the notebook.
- `consumer_complaint_eda.ipynb`: executed notebook with data quality checks, text-length analysis, product distributions, monthly drift, redaction intensity, distinctive vocabulary by product, and final validation.
- `cleaned_consumer_complaints.csv`: notebook output, retaining all 16,883 rows and seven columns.

The notebook normalizes repeated and leading or trailing whitespace in 10,115 narratives. Wording, capitalization, punctuation, numbers, and redactions are preserved. There are no models, embeddings, or train/test splits; the vocabulary section reports a log-odds ratio with an informative Dirichlet prior, which is a descriptive statistic rather than a fitted model.

Four findings bear on how the data is used later. Median complaint length varies only 1.5x across the eight categories, so length is not a shortcut to the label. The product mix shifts over the window, so a split by date is more honest than a random one. Redaction markers are 6.7% of all words, so `X{2,}` should be stripped before any vocabulary is built. And the distinctive terms per category are semantically coherent, which is the clearest evidence that the routing task is learnable from the narrative alone.

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

The source is the [official CFPB narrative archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/). The input here is a prepared subset, not the full Wells Fargo export. It was built from archive exports 5 through 15, covering September 2023 through February 2026, restricted to complaints received on or after January 1, 2024.

Before this notebook, preparation retained products with at least 100 original narratives, removed duplicate normalized text and groups with conflicting product labels, and excluded text matching the supplied JPMorgan and Bank of America datasets. The eight retained categories are checking or savings accounts, credit cards, credit reporting, money transfers or services, mortgages, debt collection, vehicle loans or leases, and payday, title or personal loans. Original CFPB label strings are retained in the CSV.

Preparation began from 23,323 Wells Fargo narratives across the archive files. Restricting to the January 2024 through February 2026 window left 17,346. Removing three categories below the 100-narrative floor left 17,271; removing nine conflicting-label groups left 17,247; removing 258 duplicate normalized narratives left 16,989; and removing 106 narratives matching the 22,465-row JPMorgan file or the 16,407-row Bank of America file left the final 16,883. Matching collapsed whitespace and ignored case.

Near-duplicate wording was not covered by this comparison, and the Morgan Stanley dataset was not available for cross-checking. These earlier selection steps are not repeated in this notebook. Repeat cross-company checks when combining different or updated team datasets. Product labels are consumer-selected categories, and the category proportions describe this selected dataset.
