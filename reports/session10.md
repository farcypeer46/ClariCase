---
team: ClariCase
session: 10
date: 2026-11-10
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
  previous: 65.9%
---

## Shipped this week
- **Structured team descriptions for Jev** (`src/jev_model/config.py`, branch `week_10`). Following TypeSafe's documented advice for options a model keeps confusing, the seven teams Jev confused most (Credit Reporting, Debt Collection, Banking Accounts, Money Transfers, Personal & Short-Term Loans, Prepaid Card, Debt & Credit Management) are now described with what the team covers, what belongs to a neighbouring team instead, and three example complaints. The other four teams and the instructions are unchanged, so the effect is attributable to the descriptions alone.
- **Example selection from the training split only** (`src/jev_model/examples.py`). Examples carry the team's label and name its core product (for instance "collection agency" for Debt Collection, "gift card" for Prepaid Card), are PII-masked, and are saved to a reviewable file so the question is identical on every run.
- **A third version with extra examples and yes/no checks** ("rich"): five examples for the four weakest teams, plus five yes/no questions asked in the same request, and a script (`src/jev_model/stack.py`) that combines them with the team answer on validation only.
- **Free analyses on cached answers:** alternative confidence scores, Jev as a second opinion to Improved Model 1, per-team thresholds, and a paired significance test.

## What changed since session 09
- **Jev improved significantly with structured descriptions:** accuracy 0.804 → 0.826 and macro-F1 0.681 → 0.719 on the same 2,000 test complaints.
- **Jev's ceiling is now well understood.** Every further approach (more examples, yes/no checks, rescoring, combining with Improved Model 1, per-team thresholds) left it short of 95% routing precision.
- **The served model is unchanged.** The app still runs Improved Model 1 alone. The north star is reported on the same 2,000-complaint test sample as in session 09 (65.9% at 95.7%), so it is unchanged this week.

## User evidence
- **Raw artifact**: `src/jev_model/`, `docs/metrics/jev_model/structured/`, `docs/metrics/jev_model/rich/`

## Metrics snapshot
- **Autonomous routing rate @95% precision: 65.9%, unchanged** (Improved Model 1 on the 2,000-complaint test sample, threshold fitted on the 2,000-complaint validation sample, as in session 09).
- **Jev's best north star: 57.4% at 94.7% precision** (structured descriptions, at its strictest confidence of 1.00), up from 50.1% at 94.4% zero-shot.
- Three Jev runs on `jev-1.14.0`, all on the same 2,000 validation and 2,000 test complaints: about $1.57 in total at TypeSafe's published price ($0.24 zero-shot, $0.58 structured, $0.75 rich).

| Test sample (2,000 complaints) | Jev zero-shot | **Jev structured** | Jev rich | Improved Model 1 |
|---|---:|---:|---:|---:|
| Accuracy | 0.804 | **0.826** | 0.826 | 0.849 |
| Macro-F1 | 0.681 | **0.719** | 0.719 | 0.775 |
| Weighted-F1 | 0.809 | **0.827** | 0.827 | 0.847 |
| North star (auto-routed at 95% precision) | 50.1% (at 94.4%)* | **57.4% (at 94.7%)*** | 58.0% (at 94.4%)* | 65.9% (at 95.7%) |
| Auto-routed at 90% precision | 74.3% | **80.5%** | 80.2% | 85.3% |
| Calibration error (ECE) | 0.112 | 0.103 | 0.104 | 0.036 |

\* Jev's strictest available threshold (confidence = 1.00); no threshold reaches 95% on validation.

- **The structured gain is statistically significant.** On the same 2,000 complaints, the structured version corrected 78 answers the zero-shot version got wrong and broke 35 (exact McNemar test, p = 0.0001). The macro-F1 gain of +0.038 has a 95% bootstrap interval of +0.019 to +0.057.

| Team (test sample) | Jev zero-shot F1 | Jev structured F1 | Improved Model 1 F1 | Support |
|---|---:|---:|---:|---:|
| T06 Mortgage | 0.891 | **0.917** | 0.870 | 65 |
| T01 Credit Reporting | 0.882 | 0.893 | 0.911 | 1,067 |
| T03 Credit Card | 0.753 | 0.778 | 0.790 | 168 |
| T04 Banking Accounts | 0.721 | **0.776** | 0.772 | 186 |
| T02 Debt Collection | 0.731 | 0.735 | 0.766 | 270 |
| T08 Vehicle Finance | 0.705 | 0.731 | 0.752 | 59 |
| T05 Money Transfers & Digital Currency | 0.688 | 0.716 | 0.723 | 110 |
| T07 Student Loan | 0.700 | 0.711 | 0.831 | 29 |
| T09 Personal & Short-Term Loans | 0.571 | 0.633 | 0.800 | 24 |
| T10 Prepaid Card | 0.538 | 0.519 | 0.583 | 15 |
| T11 Debt & Credit Management | 0.312 | 0.500 | 0.727 | 7 |

- Structured Jev now beats Improved Model 1 on Mortgage and Banking Accounts. Banking complaints sent to Money Transfers fell from 44 to 26, and over-prediction of Debt & Credit Management halved (25 predictions to 13, against 7 true cases). Teams with fewer than 30 test complaints have noisy figures.
- **Debt Collection sent to Credit Reporting did not improve (64 → 68).** 91% of these complaints mention a credit report or bureau, against 51% of Debt Collection complaints Jev routes correctly, and Improved Model 1 also misroutes 63% of them. The confusion reflects how consumers label these complaints, not a weakness specific to Jev.

## What did not work
- **Jev still cannot reach 95% precision.** Its most confident group (confidence 1.00, 57% of complaints) is 94.7% correct on test and 94.4% on validation, and cannot be split further: every answer at 1.00 gives 0.00 to all other teams.
- **Alternative confidence scores.** The top-to-second probability ratio suggested in TypeSafe's documentation, and the top-minus-second margin, give exactly the same result as the top probability, for the reason above.
- **More examples for weak teams.** Giving Student Loan, Personal & Short-Term Loans, Prepaid Card and Debt & Credit Management five examples instead of three (or a one-line description) changed only 19 of 2,000 answers; accuracy and macro-F1 were identical. The gain came from adding boundaries and examples at all, not from adding more.
- **Yes/no checks.** Five narrow yes/no questions asked alongside the team question carried only a weak signal about Jev's mistakes. Combined with the team answer on validation, the score reached 95% there but not on test (46.4% routed at 94.3%), routing fewer complaints than Jev's plain 1.00 group.
- **Jev as a second opinion.** Even the structured version adds nothing to Improved Model 1: the best combined rule ignores Jev and routes the same 65.9% as Improved Model 1 alone.
- **Per-team thresholds.** Only three teams could be given a threshold on the 2,000-complaint sample, and the routed precision did not hold on test (Banking Accounts 89%). Overall 51.9% at 93.7%, below a single shared threshold.

## Challenges / blockers
- **Label ambiguity limits every model.** The Debt Collection / Credit Reporting overlap persists across Jev and Improved Model 1, consistent with the 568,050 identical complaints filed under different categories at dataset construction.
- **Sample size for small teams.** Four teams have 7–29 test complaints in the 2,000 sample, so their per-team figures move with a handful of answers.
## Next week's goal
  - Try JevAI for issue classification 
  - Explore Jev as a fast first pass in front of the agentic issue-classification pipeline, escalating only uncertain complaints to the larger model.
  - Carried over: the full-data ModernBERT run (Sriramm) and the fully trained sequence-to-sequence model integrated with the agentic pipeline (Chaitanya).

## Individual contributions
- **Sukriti Srivastava (Product)** — Researched TypeSafe's guidance on steering Jev without fine-tuning and implemented it: structured team descriptions with boundaries and training-set examples, question versioning so cached answers are never mixed across versions, and a third version with extra examples and yes/no checks. Ran the structured and rich evaluations (about $1.33), plus free analyses: alternative confidence scores, second opinion, per-team thresholds, a paired significance test, and an investigation of the Debt Collection / Credit Reporting confusion. 24 tests.
- **Chaitanya Bagul (Engineering)** — Ported the Improved Model 3 training loop (FLAN-T5-base + LoRA team classifier) to Google Colab GPU so it can train on a free T4 instead of CPU. Built a reproducible notebook `notebooks/colab_train_improved_model_3.ipynb` that mounts Drive for persistent checkpoints/logs, uploads the two parquet splits on first run, auto-tunes `PER_DEVICE_BATCH_SIZE`/`GRAD_ACCUM_STEPS` from detected VRAM (T4/L4/A100), and streams the training log through `tee` into Drive so a runtime disconnect never loses progress. Also pinned the dependency set (uninstalled Colab's preinstalled torchao 0.10.0 since PEFT's LoRA dispatcher refuses versions below 0.16.0, which was blocking the first launch). Ran training at `--per-team-cap 5000` and evaluated two checkpoints on the same 2,000-row val subsample: checkpoint-4000 → accuracy 0.720, macro-F1 0.547, weighted-F1 0.734; checkpoint-4800 → accuracy 0.740, macro-F1 0.555, weighted-F1 0.755. Later checkpoint improves on all three metrics (+2.0 pts accuracy, +0.8 pts macro-F1, +2.1 pts weighted-F1) with `unk_rate=0.0` on both — constrained decoding kept every prediction inside the valid team set.
- **Salman Farcy (Data and Evaluation)** — **TODO (Salman): add this session's work.**
- **Sriramm S S (Users and Research)** — **TODO (Sriramm): add this session's work.**

## Lean canvas changes (if any)
