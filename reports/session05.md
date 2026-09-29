---
team: ClariCase
session: 05
date: 2026-09-29
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
  value: 46.5%
  previous: 26.4%
---

## Shipped this week
- Routing dataset v1.0: data/processed/. 1,044,615 complaints received 2024-01-01 to 2025-12-31 from all 3,801 companies in the CFPB narrative archive, not one bank. 73 product-specific issue labels mapping deterministically to 11 support teams. Construction rules and archive hashes in dataset_metadata.json; all ten checks in validation.json pass.
- Baseline 2, the team routing model: src/baseline_model_2/. TF-IDF into a calibrated Multinomial Naive Bayes over `complaint_text`, predicting one of 11 `team_id` values. Metrics, calibration curve, per-team routing table and test predictions committed to docs/metrics/baseline_model_2/. Write-up in docs/baseline_model_2.md.
- Temporal split with a real evaluation mix. Train is 2024-01-01 to 2025-06-30 capped at 3,000 per team (32,599 records); validation is July–September 2025 and test is October–December 2025, both at the true team distribution (153,502 and 109,834). Leakage groups spanning a period boundary were dropped entirely, removing 6,867 records across 1,949 groups.
- Complaint intake app: streamlit_app.py plus src/app/. A consumer types a complaint in plain words, the model routes it to a team, and they get a tracking ID to check it later. No product or issue dropdown. Storage is Supabase or local SQLite via src/app/storage.py.
- Repository reorganised. `src/models/baseline.py` and `docs/metrics/baseline/` became `src/baseline_model_1/` and `docs/metrics/baseline_model_1/`, so the two baselines and their metrics no longer share a name.
- Per-bank EDA, finished but unmerged. Wells Fargo, 16,883 complaints across 8 products (`datawellsfargo/eda`); Bank of America, 16,407 across 8 products (`databofa/eda`).

## What changed since session 04
- Corpus: 22,465 JPMorgan complaints became 1,044,615 across 3,801 companies, a 46x increase. The company-confound risk we flagged last week is now testable.
- Task: predicting a consumer's Product dropdown became routing to one of 11 support teams. Issue labels are available at 73 classes for a future model; team is a deterministic lookup.
- Evaluation: a 6,382-row single-bank test set became 109,834 rows at the real team mix, with validation held separately for fitting thresholds.
- Calibration reversed sign. Last week wrapping the SVM cost 0.005 macro-F1; this week isotonic calibration gains 0.074 and cuts ECE from 0.0664 to 0.0259.
- There is now a product. Session 04 had no interface at all.

## User evidence
- The app exists and runs, but it has not been put in front of anyone yet. No consumer or ops user has used it, so there is still no user evidence this session.
- **Raw artifact**: streamlit_app.py, src/app/storage.py, docs/baseline_model_2.md

## Metrics snapshot
- **Autonomous routing rate @95% precision: 46.5%, up from 26.4%.** 51,034 of 109,834 test complaints routed without a human at a 0.872 confidence threshold.
- The 95% target was not actually met on test. Realised precision is 94.44% (95% CI 94.24–94.65). The threshold was fitted on validation, where it gave 48.18% coverage; on test it delivers 46.46% and lands just under the precision target. The same undershoot appears at the other operating points: 89.68% at the 90% threshold, 84.31% at the 85% threshold. Thresholds fitted on July–September do not quite hold in October–December.

- The model was evaluated two ways. The production figure uses the real team mix, which is what an ops queue actually receives. The balanced figure samples the test set evenly across teams, which measures the classifier's ability independently of how skewed the incoming traffic is.

| Test protocol | n | Accuracy | Macro-F1 | Weighted-F1 | Dummy accuracy | Dummy macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Real team mix, Oct–Dec 2025 | 109,834 | 0.800 | 0.647 | 0.793 | 0.527 | 0.063 |
| Balanced, ~600 per team | 6,601 | 0.730 | **0.732** | 0.729 | 0.085 | 0.014 |

- The two tell a consistent story. On real traffic, accuracy is high (0.800) because 53% of complaints are Credit Reporting, but macro-F1 drops to 0.647 as the small teams drag it down. On a balanced set, accuracy falls to 0.730 while macro-F1 rises to 0.732, and the dummy collapses from 0.527 to 0.085. The model is genuinely reading the text rather than exploiting the prior; the real-mix accuracy is flattered by imbalance and the balanced macro-F1 is the cleaner measure of the classifier itself.

| Team | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| T07 Student Loan | 0.898 | 0.881 | 0.889 | 570 |
| T06 Mortgage | 0.825 | 0.896 | 0.859 | 565 |
| T10 Prepaid Card | 0.811 | 0.782 | 0.796 | 559 |
| T08 Vehicle Finance | 0.807 | 0.737 | 0.770 | 585 |
| T05 Money Transfers & Digital Currency | 0.746 | 0.726 | 0.736 | 649 |
| T01 Credit Reporting | 0.667 | 0.780 | 0.719 | 619 |
| T09 Personal & Short-Term Loans | 0.693 | 0.726 | 0.709 | 632 |
| T03 Credit Card | 0.715 | 0.633 | 0.672 | 603 |
| T04 Banking Accounts | 0.578 | 0.752 | 0.653 | 614 |
| T02 Debt Collection | 0.654 | 0.631 | 0.642 | 613 |
| T11 Debt & Credit Management | 0.731 | 0.510 | 0.601 | 592 |

- Per class on the balanced set, performance splits cleanly by how distinctive the vocabulary is. The specialist teams do best: Student Loan 0.889 and Mortgage 0.859. The generalist and overlapping teams do worst: Debt & Credit Management 0.601 at 0.510 recall, Debt Collection 0.642, and Banking Accounts 0.653 at only 0.578 precision. Credit Reporting, despite being the largest class by far, manages 0.719, so size does not buy separability.

- Calibration helps this model. Uncalibrated test accuracy is 0.7586 and macro-F1 0.5737; isotonic calibration raises them to 0.7999 and 0.6475, and cuts ECE from 0.0664 to 0.0259.
- Per-team precision at the routing threshold ranges from 0.9905 (Money Transfers) to 0.8230 (Personal & Short-Term Loans). Ten of 11 teams clear the 0.85 precision floor; T09 does not.
- Auto-routed share is very uneven. Credit Reporting sends 67.1% of its complaints straight through, Mortgage 56.9%, but Money Transfers only 15.7% and Debt Collection 17.5%. The headline 46.5% is carried by the largest class.
- Accuracy is the wrong headline on real traffic. Credit Reporting is 61.8% of the corpus and 52.7% of the test window, so always guessing it scores 0.527. Macro-F1 is the honest number in both protocols.
- Is this the same model that is running in the product? Yes. streamlit_app.py loads `src/baseline_model_2/predict.py`, the same artifact the metrics above were computed from.

## What did not work
- The north star target is still unmet. 46.5% is measured at 94.44% precision, not 95%, so the headline number is slightly optimistic against its own definition. Better than the 26.4% it replaces, but more than half of complaints still need a human.
- T09, Personal & Short-Term Loans, routes at 0.823 precision, below the 0.85 guardrail. It is a small team, 1,640 test complaints, and only 305 are auto-routed.
- Every operating point undershoots its validation estimate by roughly half a point of precision and 1.5–2 points of coverage. Team mix moves across the corpus: comparing pre-July-2025 with October–December, Credit Reporting falls 10.0 points (63.1% to 53.1%) while Debt Collection rises 3.6 and Credit Card 2.4. Thresholds frozen on one quarter do not transfer cleanly to the next.
- Training uses 32,599 of 1,044,615 records, capped at 3,000 per team. The cap protects small teams, but 97% of the corpus is unused and we have not tested whether more data helps the large classes.
- 568,050 narrative candidates were excluded during dataset construction because identical text carried conflicting product-issue labels, and 415,097 more as duplicates. Last week we wrote that label quality was unmeasured; this is the measurement, and over half the delivered dataset's worth of records was discarded as ambiguous.
- The app has not been deployed to a public URL, so nothing in the README points at a running instance.
- Severe imbalance at both granularities: 178:1 across teams, 627:1 across the 73 issue labels. One issue label is 31.1% of the corpus on its own.
- Debt & Credit Management recalls only 51.0% on the balanced set, and Credit Card 63.3%. Nearly half of T11's complaints are missed, most likely to Debt Collection and Credit Reporting, which share credit-repair and dispute vocabulary. Banking Accounts has the opposite failure, 0.752 recall on 0.578 precision, so it absorbs complaints that belong elsewhere.
- The balanced evaluation is not committed to the repository. Only the real-mix run is in docs/metrics/baseline_model_2/, so the per-class table above cannot currently be reproduced from a clean clone.

## Challenges / blockers
- No user contact yet. The app was finished at the end of the week, so there was no time to put it in front of anyone. This is the third session without user evidence.
- Not publicly deployed. It runs locally; a hosted URL is needed before a stranger can test it.
- Threshold transfer across time is unsolved. Fitting on validation and freezing for test is the right protocol, but the drift means the guarantee is soft. Either refit more often or widen the margin.
- The 1.91 GB dataset CSV is not in the repository and `data/processed/` is gitignored, so the model cannot be retrained from a clean clone even though its metrics are now committed.
- The 73-label issue model has not been attempted. Team routing is the coarser problem; issue-level prediction is where the dataset's granularity actually pays off.

## Next week's goal
<!-- Sukriti to write and commit this section. -->

## Individual contributions
- Sukriti Srivastava (Product): Streamlit complaint intake app, complaint storage layer, and the model rebuild behind it (PR #18)
- Chaitanya Bagul (Engineering): Routing dataset construction and Baseline 2, the calibrated team routing classifier with its metrics and write-up (PR #16, #17)
- Salman Farcy (Data and Evaluation): Wells Fargo corpus and EDA; repository restructure aligning both EDA branches to the main layout
- Sriramm S S (Users and Research): Bank of America corpus and EDA

## Lean canvas changes (if any)
- The archive is a closed set. The CFPB stopped publishing narratives, and this dataset is built entirely from the 21 archived export files. The corpus cannot grow from this source, only be re-cut. Any live product will eventually run on inputs drawn from a different distribution than it was trained on.
- Scope widened from one bank to an industry corpus. The product is no longer "route complaints for a bank" but "route complaints across 3,801 companies into 11 support functions", which changes who the buyer is.
- The consumer is now a user, not just a data source. The app removes the product and issue dropdowns and replaces them with free text plus a tracking ID, so the value proposition now includes the person filing the complaint, not only the ops team reading it.
