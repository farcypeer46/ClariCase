---
team: ClariCase
session: 05
date: 2026-09-28
members:
  - name: Sukriti Srivastava
    github: Sukriti-124
    hat: Product
  - name: Chaitanya Bagul
    github: elderflamevandle
    hat: Engineering
  - name: Salman Farcy
    github: farcypeer46
    hat:  Data&Eval 
  - name: Sriramm S S 
    github: SriRammSS
    hat: Users&Research 
north_star:
  metric: Autonomous routing rate at >=95% precision
  value: Not re-measured this week
  previous: 26.4%
---

## Shipped this week
- Routing dataset v1.0: data/processed/. 1,044,615 complaints received 2024-01-01 to 2025-12-31, drawn from all 3,801 companies in the official CFPB narrative archives, not one bank. 73 product-specific issue labels mapping deterministically to 11 support teams. Construction rules, archive hashes and reconciliation counts are recorded in dataset_metadata.json; all ten checks in validation.json pass.
- Target changed. Last week we predicted Product directly, 11 classes. We now predict `product_issue_id`, 73 classes such as `Credit card :: Fees or interest`, and obtain the team by lookup through issue_mapping.csv. Routing became a deterministic join rather than a second model.
- Team classifier baseline: src/baseline_model_1/. TF-IDF into Multinomial Naive Bayes over `complaint_text`, predicting `team_id`, with a majority-class dummy for comparison. Splits are 60/20/20 and grouped by `leakage_group_id` so near-identical narratives cannot straddle train and test.
- Per-bank EDA, on branches and not merged. Wells Fargo, 16,883 complaints across 8 products (`datawellsfargo/eda`); Bank of America, 16,407 across 8 products (`databofa/eda`). Both restructured to match main's layout with bank-prefixed filenames.
- Still not deployed. There is no running product, so nothing is live.

## What changed since session 04
- Corpus: 22,465 JPMorgan complaints became 1,044,615 across 3,801 companies, a 46x increase. Last week's numbers described one bank's product mix; the company-confound question we raised is now testable.
- Label granularity: 11 product classes became 73 product-issue classes, then an 11-team routing lookup.
- Leakage control moved from a plain temporal split to grouped splits on `leakage_group_id`.
- Corpus window is now fixed at two complete calendar years, 2024 and 2025, instead of running to 2026-01.

## User evidence
- No user contact again this week. The work was dataset construction and a second baseline. Deployment has now slipped two sessions.
- **Raw artifact**: data/processed/README.md, data/processed/validation.json, data/processed/dataset_metadata.json

## Metrics snapshot
- Autonomous routing rate @95% precision: not re-measured. The number below is last week's and refers to a different task.
- Baselines any model on the new corpus must beat, computed from issue_mapping.csv and team_coverage.csv:

| Task | Classes | Majority-class accuracy | Majority-class macro-F1 | Imbalance |
|---|---|---|---|---|
| Team routing | 11 | **0.6176** | 0.0694 | 178:1 |
| Product-issue | 73 | 0.3105 | 0.0065 | 627:1 |

- The Credit Reporting Team alone is 61.8% of the corpus, 645,108 of 1,044,615 records. A model that always predicts it is right 61.8% of the time, so accuracy is close to meaningless here and macro-F1 is the only honest headline.
- Class sizes at the 73-label target run from 324,392 (`Credit reporting :: Incorrect information on your report`, 31.1% of the dataset on its own) down to 517, with a median of 3,274.
- Team mix drifts over the corpus window. Comparing the pre-July-2025 records with October–December 2025: Credit Reporting falls 10.0 points, 63.1% to 53.1%, while Debt Collection rises 3.6, Credit Card 2.4 and Banking Accounts 2.1. A random split would hide this; the grouped split does not account for it either, since it groups by near-duplicate text rather than by date.
- Previous: 26.4% at >=95% precision, ECE 0.0408, on 6,382 JPMorgan test complaints predicting 7 product classes. Session 04 reported 11 product categories in the corpus, but four fell below the class floor and the committed classification report covers seven.
- The new team classifier writes metrics.json, classification_report.json, confusion_matrix.csv and predictions_test.csv to src/baseline_model_1/artifacts/, but that path is gitignored, so no measurement from it is in the repository.
- Is this the same model that is running in the product? There is still no product. Nothing is deployed.
- The two numbers are not comparable in any case: 7 product classes on one bank versus 11 teams on 3,801 companies, from different corpora and different splits.
- Two versions of last week's model are committed in different places. `docs/metrics/baseline/` on main holds the calibrated run, macro-F1 0.7264 and accuracy 0.8068; the `baseline_model_1` branch holds the uncalibrated run at `reports/baseline/`, macro-F1 0.7309 and accuracy 0.7991. Same test set, same dummy row. Only the calibrated one was reported.

## What did not work
- 568,050 narrative candidates were excluded because identical text carried conflicting product-issue labels. That is more than half the size of the delivered dataset, discarded as ambiguous supervision. Last week we wrote that label quality was unmeasured; this is the first measurement of it, and it is worse than we assumed.
- 415,097 further records were dropped as duplicate narratives, and 7,884 for insufficient issue support. Of 8,177,233 source rows in range, 6,140,798 had no narrative at all.
- No metrics are committed for the new model, so this week's work cannot be evaluated from the repository by anyone who has not run it locally.
- The 1.91 GB dataset CSV is not in the repository and `data/processed/` is now gitignored, while the small mapping files inside it remain tracked. Anyone cloning fresh gets the mappings but not the data the baseline needs.
- The baseline samples 3,000 complaints per team, roughly 33,000 of 1,044,615 rows. It is a smoke test of the pipeline, not a measurement of what the corpus supports.
- The two per-bank EDAs are finished but sit unmerged on branches, so they are not visible from main.

## Challenges / blockers
- Still no deployed app. This was the top blocker in session 04 and nothing shipped against it. Every user-evidence item remains empty until a stranger can open a URL.
- The north star has not moved because it has not been measured. We cannot say whether the larger corpus helped, hurt, or changed nothing.
- Reproducibility gap. Metrics and splits are gitignored and the full CSV is absent, so no result in this repository can currently be reproduced from a clean clone.
- Task definition shifted mid-project. Session 04 measured product classification; session 05 builds team routing over issue labels. Prior numbers are now historical, not a baseline to beat.
- Severe class imbalance at both granularities, 178:1 across teams and 627:1 across issue labels. One team is 61.8% of the corpus and one issue label is 31.1%, so the majority-class floor is high and macro-F1 will be dominated by the long tail. Whether to reweight, resample or accept it is undecided.
- The front end does not exist yet. The routing model has no interface, so there is still nothing a user can open, which is why the user-evidence section is empty for the second session running.

## Next week's goal
<!-- Sukriti to write and commit this section. -->

## Individual contributions
- Sukriti Srivastava (Product): Session reporting
- Chaitanya Bagul (Engineering): Routing dataset construction and the TF-IDF + Multinomial Naive Bayes team classifier (PR #16)
- Salman Farcy (Data and Evaluation): Wells Fargo corpus and EDA; repository restructure aligning both EDA branches to the main layout
- Sriramm S S (Users and Research): Bank of America corpus and EDA

## Lean canvas changes (if any)
- The archive is the only remaining source. Last week we noted the CFPB stopped publishing narratives. This week's dataset is built entirely from the 21 archived export files, which are a closed set. Our corpus cannot grow from this source; it can only be re-cut.
- Scope widened from one bank to an industry corpus. The product is no longer "route complaints for a bank" but "route complaints across 3,801 companies into 11 support functions", which changes who the buyer is and should be reflected in the canvas.
