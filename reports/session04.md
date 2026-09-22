---
team: ClariCase
session: 04
date: 2026-09-22
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
  value: 26.4%
  previous: N/A (first measurement)
---

## Shipped this week
- Data pipeline: src/data/{ingest,explore,prepare}.py. Loads the CFPB export, writes a profile to docs/data_profile.md, and produces a cleaned temporal split through a single entry point, get_splits(). 22,465 raw rows to 22,096 after cleaning; 15,714 train / 6,382 test.
- Baseline model: src/models/baseline.py. Majority-class dummy plus TF-IDF (1,2)-grams into a calibrated LinearSVC. All metrics written to docs/metrics/baseline/.
- Not deployed. There is no running product yet, so nothing is live

## User evidence
- None captured this week. No real user has touched a running product, because there is no running product.
- **Raw artifact**: docs/data_profile.md, docs/metrics/baseline/



## Metrics snapshot
- Autonomous routing rate @95% precision: 26.4% (First measurement)
- Measured on: temporal held-out set, 2025-07-01 to 2026-01-30, 6,382 complaints, deduplicated before splitting
- Is this the same model that is running in the product? No because there is no live product yet.

## What did not work
- 26.4% is not a useful product. Hitting 95% precision requires a 0.90 confidence threshold, which leaves three quarters of the queue for a human. An ops team saving a quarter of their reading time is a weak pitch.
- Debt collection recall is 0.41. Well over half get missed, mostly to credit reporting — the two share FCRA and FDCPA vocabulary almost entirely.
- Money transfer misroutes to Checking or savings 41% of the time. A Zelle dispute reads exactly like a checking complaint because the money left a checking account. We do not yet know whether this boundary is learnable or genuinely ambiguous.
- Calibration costs accuracy. Macro-F1 falls 0.731 to 0.726 when the SVM is wrapped for probability estimates. Small, but we are reporting the lower number because it belongs to the model that can actually produce confidence scores.
- Our label quality is unmeasured. Product labels are chosen by the consumer from a dropdown at intake, never audited. Some share of our 19% error rate is the data being wrong, and we cannot currently say how much.

## Challenges / blockers
- No deployed app. Everything else depends on this. Nothing can be tested with a user until a stranger can open a URL.
- The task is partly circular as posed. Every complaint in this corpus arrived through the CFPB web form already labelled by the consumer, so predicting the label on this data proves little. The product only makes sense as: learn the taxonomy here, apply it to unlabelled channels elsewhere.
- Two classes are thin in test - Mortgage (157) and Vehicle loan (104). Per-class recall on these will swing between runs and should be read as noisy, not precise.
## Next week's goal
- Widen the corpus. Add Bank of America and Wells Fargo to the current JPMorgan-only data. Today's numbers describe one bank's product mix, so the class floor and the config thresholds get re-checked after the merge, more data may pull Prepaid card or Payday loan back above the cutoff we dropped them at. We will also check whether per-class performance differs by company, since a model can learn "sounds like Wells Fargo, therefore credit card" instead of learning the language of card complaints.
- Move the north star. 26.4% is the number to beat. Three levers, cheapest first: proper calibration (temperature scaling rather than Platt, measured by ECE), then per-class thresholds instead of one global cut, then a fine-tuned transformer. The two worst classes - debt collection at 0.41 recall and money transfer at 0.50 are where the headroom is, since low-confidence predictions concentrate there.

## Individual contributions
- Sukriti Srivastava (Product): Data ingestion and exploration
- Chaitanya Bagul (Engineering): Baseline Modelling 
- Salman Farcy (Data and Evaluation): Data cleaning  
- Sriramm S S (Users and Research): Data collection 

## Lean canvas changes (if any)
- New data risk. The CFPB stopped publishing complaint narratives in August 2026 and removed the consent mechanism in September. Our corpus is a fixed historical snapshot, not a live feed - the "updates daily" claim is gone. It also strengthens the case for the product: the labelled signal we trained on is no longer being generated.
