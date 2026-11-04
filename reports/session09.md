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

### Issue-first fine-tuning — Sriramm

The objective for these experiments is **complaint text → product-specific issue → team**. The original 73 issue labels are preserved, and each issue maps to one of the 11 product teams through a fixed lookup. Only the complaint narrative is supplied as input; the recorded product, issue and team are not input features. This is a separate experiment from the direct team classifiers above and from Improved Model 3 (FLAN-T5).

**Earlier Llama pilot.** Fine-tuned Llama 3.2 3B Instruct in Colab using a 4-bit base model and LoRA adapters. Changed the generated target from an opaque issue ID to a descriptive `Product :: Issue` label, then mapped the generated label back to its issue and team. Training used answer-only loss, with checks that the prompt and padding tokens were masked. The completed one-epoch pilot used 7,300 training complaints. On a class-balanced validation diagnostic of 730 complaints, it achieved **40.68% issue accuracy, 0.3929 issue macro-F1 and 74.79% team accuracy**, with 100% valid labels. These are validation results, not final test results.

**Completed Zaratan comparison.** Fine-tuned ModernBERT-base and DeBERTa-v3-base with a 73-class classification head, updating the encoder weights as well as the head. Both used the same 29,141 training complaints, capped at 400 per issue where available, for two epochs. Settings included a learning rate of 2e-5, effective batch size 32, seed 20261102 and a 512-token limit with beginning-and-end cropping. The training window was January 2024–June 2025; validation covered July–September 2025 (3,650 complaints, 50 per issue), and test covered October–December 2025 (2,190 complaints, 30 per issue). Known audit groups spanning time splits were excluded without changing the original labels.

| Model | Validation issue macro-F1 | Test issue accuracy | Test issue macro-F1 | Test team accuracy | Test team macro-F1 |
|---|---:|---:|---:|---:|---:|
| ModernBERT-base | **0.3586** | **36.94%** | **0.3592** | 68.90% | 0.6835 |
| DeBERTa-v3-base | 0.2556 | 29.27% | 0.2503 | **70.37%** | **0.6912** |

Each model's checkpoint was selected by its lowest validation loss. ModernBERT was selected for the larger run because it had the higher **validation issue macro-F1**, which matches the issue-first objective. DeBERTa had better team accuracy but was less accurate at identifying the specific issue. Both produced valid labels for every evaluated complaint, and the metrics were recomputed from saved predictions. Each pilot took about six minutes of training on the cluster's H100 GPU, excluding setup and queue time.

**Full training status, November 2:** prepared all **720,088 eligible training complaints**, with no per-class cap. This is the original 725,654-row training partition minus 5,566 rows belonging to known cross-period audit groups; validation and test remain separate. Submitted a fresh ModernBERT run for two epochs, retaining the same validation/test diagnostics and saving resumable checkpoints. Job **23738671 is pending GPU allocation** on the A100 partition; no full-data results are available yet. The queued Qwen pilot was cancelled to focus on this run.

**Interpretation:** the remaining challenge is distinguishing issues within and across products, rather than merely producing a valid label. These scores measure agreement with the existing dataset labels. The evaluation samples are balanced by issue, the historical partitions have been examined previously, and the earlier Llama pilot used different samples and settings. Consequently, these figures do not establish that one model beats Llama, Jev or the served classifier. No confidence threshold or autonomous routing rate at 95% precision has been established for these issue-first models, and none has replaced the deployed model.

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
  - **Issue-first fine-tuning (Sriramm):** complete the queued full-data ModernBERT run, evaluate it on the fixed diagnostics, and inspect per-issue errors before considering confidence calibration or deployment.
  - Train Improved Model 3 on the full training set with class balancing, and score it on the test set with calibration (carried over).
  - Try Jev with a few training examples per team (about $0.90 on the same sample) and on a larger test sample (about $0.50), if the team approves the spend.
  - **Fully trained Seq-2-seq model (Chaitanya)** Fully finetune the Seq-2-seq model for classsification and integrate it in the agentic pipeline for Issue clasification. Along with it , add better prompt for issue clasification so that it is able to classify issue correctly without getting confused.
  

## Individual contributions
- **Sukriti Srivastava (Product)** — Built the Jev evaluation pipeline: API client following TypeSafe's reference, PII-masked inputs, per-complaint answer cache, cost estimate with explicit confirmation, offline mock mode and 15 tests. Ran the zero-shot evaluation against Improved Model 1 on the same samples, plus the second-opinion and calibration analyses. (branch `jev_ai`)
- **Chaitanya Bagul (Engineering)** — Build an agentic ai based pipeline by leveraing the capabilities og GPT-OSS 120B model from OpenAI on Groq Cloud platform to classify issues correctly. Parallely finetuning seq-2-seq model for team classification and it is taking lot of time in traininig on local GPUs, so traininchunks and saving intermediate checkpoints to resume traing later on. (branch `improved_model_3`) 
- **Salman Farcy (Data and Evaluation)** — Assessed the product's market position and value case, recorded in [`docs/market_analysis.md`](../docs/market_analysis.md). Sized the complaint-management software category at $2.9–3.1B in 2025–26, growing to $5.8–6.3B by 2033–34, and placed ClariCase against Zendesk, which acquired Forethought in March 2026, alongside Salesforce, Ada and Fini. Established that every published competitor reports accuracy alone (85–93%, vendor-reported) with no coverage figure at a precision floor, which makes the project's 61.5% auto-routed at 94.8% precision a stricter claim than any figure currently stated in this market. Set the cost baseline at $13.50 per human-handled contact, declining to attribute a share of it to triage because no published source separates that step. Recorded distribution rather than accuracy as the main competitive barrier, since the incumbents own the ticketing system the complaint already sits in.
- **Sriramm S S (Users and Research)** — Extended the Llama fine-tuning work with a matched ModernBERT/DeBERTa comparison on Zaratan for 73-issue prediction followed by routing to 11 teams. Preserved the original labels, checked temporal split separation and issue-to-team mapping, evaluated saved predictions, and selected ModernBERT using validation issue macro-F1. Prepared the full 720,088-row training split and submitted the full ModernBERT run, which is awaiting GPU allocation. Recorded the earlier descriptive-label Llama pilot and the new encoder results above. Tracked in [#26](https://github.com/farcypeer46/ClariCase/issues/26).

## Lean canvas changes (if any)

