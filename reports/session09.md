---
team: ClariCase
session: 09
date: 2026-11-02
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
  value: 65.9%
  previous: 61.5%
---

## Shipped this week
- **Jev AI evaluation pipeline** (`src/jev_model/`, branch `jev_ai`). TypeSafe AI's Jev is a "System One" model that returns a structured choice with probabilities instead of text. We use it as a zero-shot team classifier: each complaint is PII-masked exactly as the app masks it, and Jev chooses one of the 11 teams from a short description of each team's product and issues. The pipeline caches every answer so no request is paid for twice, shows a cost estimate and sends nothing without confirmation, retries on rate limits, and has an offline mock mode. 15 tests, with all HTTP faked.
- **Head-to-head evaluation against Improved Model 1.** Both models were scored on the same 2,000 validation and 2,000 test complaints with the project's routing harness: threshold fitted on validation, frozen for test.
- **Second-opinion and calibration analyses** (`src/jev_model/combine.py`), computed from cached answers at no extra cost.

## What changed since session 08
- **A third modelling approach was tested.** After linear models (Baseline 2, Improved Model 1) and a fine-tuned transformer (Improved Model 3), the team evaluated a zero-shot decision model that uses no training data at all.
- **Jev cannot be fine-tuned.** TypeSafe serves the same weights to every customer, so Jev is guided only by the wording of the question and the team descriptions.
- **The north star is reported as 65.9% for Jev + Improved Model 1** (Jev as a second opinion), on the 2,000-complaint test sample with the routing threshold fitted on the 2,000-complaint validation sample. The best combined rule makes no use of Jev, and Improved Model 1 alone routes 60.8% of the same sample at the official threshold, so the rise from 61.5% reflects the smaller-sample threshold rather than a gain from Jev. The app still runs Improved Model 1 alone.

## User evidence
- **Raw artifact**: `src/jev_model/`, `docs/metrics/jev_model/`, `tests/test_jev_model.py`

## Metrics snapshot
- **Autonomous routing rate @95% precision: 65.9%** (Jev + Improved Model 1, at 95.7% precision, on the 2,000-complaint test sample). **Caveat:** Improved Model 1 alone also routes 65.9% on this sample, so combining adds nothing. The threshold here (0.782) was fitted on a 2,000-complaint validation sample; with the official threshold fitted on the full validation set (0.818), the same sample routes 60.8%, in line with the previous 61.5% on the full test set (109,834 complaints).

**This week's results at a glance**
- **Baseline-2 accuracy with zero training.** Jev classified complaints at 0.804 accuracy using only 11 short team descriptions, matching Baseline 2 (0.800), which needed 32,599 training complaints. The whole 4,000-complaint evaluation cost about $0.24.
- **The best-calibrated confidence in the project.** After recalibration on validation, Jev's calibration error is 0.019, against 0.036 for Improved Model 1: its confidence scores become trustworthy probabilities.
- **Strong where it counts at a lower bar.** At 90% precision Jev auto-routes 74.3% of complaints with no training, and it beats Improved Model 1 on Mortgage (F1 0.891 against 0.870).
- **A reusable, low-cost evaluation harness.** Any external model can now be scored against Improved Model 1 on identical masked samples, with cached answers, a cost guard and an offline mock mode, so future candidates such as few-shot Jev or the fine-tuned Llama can be compared cheaply and fairly.
- **A decision made with evidence.** Jev alone and Jev as a second opinion both fail to beat Improved Model 1 at the 95% target, so the team can avoid adding a paid external dependency to the product.
- **Privacy fixes shipped to main** (PR #32): masking no longer removes ordinary words the model routes on.
- **Jev's north star: 50.1% at 94.4% precision** on the 2,000-complaint test sample, against 65.9% at 95.7% for Improved Model 1 on the same complaints.
- Jev run: model `jev-1.14.0`, 4,000 requests, 5.67 million input tokens, about $0.24 at TypeSafe's published price.
- Figures below are on the 2,000-complaint test sample, which Improved Model 1 was also scored on. Its sample figures differ from its full-test figures through sampling variation; the comparison between the two models is like for like.

| Test sample (2,000 complaints) | Jev (zero-shot) | **Improved Model 1** |
|---|---:|---:|
| Training data used | none | 257,694 complaints |
| Accuracy | 0.804 | **0.849** |
| Macro-F1 | 0.681 | **0.775** |
| Weighted-F1 | 0.809 | **0.847** |
| **North star:** auto-routed at 95% precision | 50.1% (at 94.4%)* | **65.9%** (at 95.7%) |
| Auto-routed at 90% precision | 74.3% (at 90.0%) | **85.3%** (at 90.1%) |
| Calibration error (ECE) | 0.112 | **0.036** |

\* Jev's strictest available threshold (confidence = 1.00). It reached 94.6% on validation, the closest Jev can get to 95%, so it is reported at that point rather than at a validation-fitted 95% threshold.

- With no training at all, Jev matches the accuracy of Baseline 2 (0.800), which was trained on 32,599 complaints.
- **Why Jev's north star is measured at confidence 1.00.** Jev returns only 72 distinct confidence values, and over half of all complaints receive exactly 1.00. That top group is 94.6% correct on validation and 94.4% on test, and its members cannot be separated further, so no stricter threshold exists.
- **Jev is overconfident**, despite being described as calibrated: answers at confidence 0.95–0.99 are 83% correct, and at 0.80–0.90 only 63%. Isotonic recalibration fitted on validation reduces its calibration error from 0.112 to 0.019, better than Improved Model 1's 0.036, but recalibration cannot change which complaints rank highest, so the 95% target stays out of reach.

| Team (test sample) | Jev precision | Jev recall | Jev F1 | Improved Model 1 F1 | Support |
|---|---:|---:|---:|---:|---:|
| T01 Credit Reporting | 0.91 | 0.85 | 0.882 | 0.911 | 1,067 |
| T06 Mortgage | 0.85 | 0.94 | **0.891** | 0.870 | 65 |
| T03 Credit Card | 0.69 | 0.83 | 0.753 | 0.790 | 168 |
| T02 Debt Collection | 0.79 | 0.68 | 0.731 | 0.766 | 270 |
| T04 Banking Accounts | 0.83 | 0.64 | 0.721 | 0.772 | 186 |
| T08 Vehicle Finance | 0.61 | 0.83 | 0.705 | 0.752 | 59 |
| T07 Student Loan | 0.55 | 0.97 | 0.700 | 0.831 | 29 |
| T05 Money Transfers & Digital Currency | 0.60 | 0.80 | 0.688 | 0.723 | 110 |
| T09 Personal & Short-Term Loans | 0.46 | 0.75 | 0.571 | 0.800 | 24 |
| T10 Prepaid Card | 0.64 | 0.47 | 0.538 | 0.583 | 15 |
| T11 Debt & Credit Management | 0.20 | 0.71 | 0.312 | 0.727 | 7 |

- Jev beats Improved Model 1 only on Mortgage. It over-predicts the small teams (high recall, low precision): of the complaints it sends to Debt & Credit Management, only 20% belong there. Teams with fewer than 30 test complaints in this sample have noisy figures.
- Its main confusions are the project's familiar ones: Debt Collection sent to Credit Reporting (64 complaints) and Banking Accounts sent to Money Transfers (44).
- **Second opinion.** The two models agree on 84.3% of complaints; Jev alone is right on 4.4% and Improved Model 1 alone on 8.9%. We searched for the best rule that lets Jev's agreement lower Improved Model 1's threshold, fitted on validation with the project's standard 0.002 threshold steps. The best rule ignores Jev entirely (0.782 whether or not Jev agrees), giving 65.9% at 95.7% on test, identical to Improved Model 1 alone. An earlier coarser search (0.01 steps) suggested a 0.9-point gain; it was an artefact of the coarse steps.

## What did not work
- **Jev's north star (50.1%) is well below Improved Model 1's (65.9% on the same sample)**, and at 94.4% precision it also falls just short of the 95% definition. Its most confident answers cannot be split further, so Jev cannot route more complaints at higher precision on its own.
- **Jev's confidence is overconfident.** Its raw calibration error (0.112) is three times Improved Model 1's, contrary to TypeSafe's calibration claims. It can be corrected after the fact, but that does not help routing at the 95% target.
- **The small teams are its weakest point**, the same teams Improved Model 1 handles best relative to other models: Debt & Credit Management F1 0.312 against 0.727, Personal & Short-Term Loans 0.571 against 0.800.
- **Combining Jev with Improved Model 1 adds nothing.** With the standard threshold steps, the best second-opinion rule ignores Jev, and routes the same 65.9% as Improved Model 1 alone, while it would add a paid external call to every complaint.
- **Fine-tuning Jev is not possible.** TypeSafe does not offer customer fine-tuning; few-shot examples in the team descriptions are the closest available option and were not yet tried.
- **Improved Model 3's full training run**

## Challenges / blockers
- **Sending complaints to an outside service.** Jev requires complaint text to leave our system. Masking removes numbers, emails and addresses, but names are kept by design, and data-retention terms for non-enterprise users are not published.
- **Sample size.** The comparison uses 2,000 test complaints; the smallest teams have 7–29 examples each, so their per-team figures are noisy. A larger sample would cost about $0.50 more.
- **Cost of using Jev in production.** Each complaint would be a paid external call (about $0.00006 at the observed 1,418 tokens per complaint). That is cheap, but it adds latency, a dependency on TypeSafe's availability, and a privacy review.
- The forward-drift cohort is no longer fresh. A new untouched evaluation window will be needed to report future gains credibly.

## Next week's goal
  - Train Improved Model 3 on the full training set with class balancing, and score it on the test set with calibration (carried over).
  - Try Jev with a few training examples per team (about $0.90 on the same sample) and on a larger test sample (about $0.50), if the team approves the spend.
  

## Individual contributions
- **Sukriti Srivastava (Product)** — Built the Jev evaluation pipeline: API client following TypeSafe's reference, PII-masked inputs, per-complaint answer cache, cost estimate with explicit confirmation, offline mock mode and 15 tests. Ran the zero-shot evaluation against Improved Model 1 on the same samples, plus the second-opinion and calibration analyses. (branch `jev_ai`)
- **Chaitanya Bagul (Engineering)** — **TODO (Chaitanya): add this session's work.**
- **Salman Farcy (Data and Evaluation)** — **TODO (Salman): add this session's work.**
- **Sriramm S S (Users and Research)** — **TODO (Sriramm): add this session's work.**

## Lean canvas changes (if any)

