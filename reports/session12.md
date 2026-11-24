---
team: ClariCase
session: 12
date: 2026-11-24
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
  metric: Autonomous routing rate at >=90% precision
  value: 86.3%
  previous: 86.3%
---

## Shipped this week
- **Human review queue** (`pages/review.py`): complaints routed below the confidence threshold are held at `/review` for an admin to read and assign to a team. The model's choice is retained alongside the admin's (`model_team_id`, `model_team_name`, `reviewed_at`, `review_note`). This closes session 11's open item on human review.
- **Admin analytics** (`pages/admin.py`, `src/app/admin_common.py`): incoming volume, routing mix (auto-routed, needs review, unknown) and team distribution, served at `/admin` behind an admin password when complaints are stored in Supabase.
- **Bulk intake** (`src/app/bulk.py`): up to 2,000 complaints per uploaded file, with a CSV export of the tracking IDs issued.
- **Content safety gate** (`src/safety/`, PR #45): screens a complaint before classification. Threats, harassment and profanity are matched deterministically against normalised tokens, including disguised spellings; claims about the outside world are assessed by an LLM judge on the agentic backend. A blocked complaint is not classified, routed or stored, and the submitter is shown the reason.
- **Product-conditioned issue service** (`platform/services/issue-api/`, PRs #42, #43): the issue classifier is now a platform service with a catalogue conditioned on the product, so only issues valid for that product can be returned. Includes contracts, Dockerfile and API tests.
- **Explanation highlighting** (`src/app/highlight.py`): the explanation phrases introduced in session 11 are now rendered within the complaint text.
- **Deployment and cost analysis** (`docs/deployment_cost.md`, PR #44).

## What changed since session 11
- **The system now has an operator surface.** Below-threshold complaints have a defined destination and owner, completing the review half of the routing policy.
- **Submissions can be refused before routing.** The safety gate introduces a fourth outcome alongside routed, under review and unknown.
- **Review decisions are recorded as labels.** Each assignment stores the model's prediction against the reviewer's, providing in-product ground truth independent of the CFPB corpus.
- **Intake accepts batches.** The application is no longer limited to one complaint per submission.
- **No modelling work this session.** The served ensemble, its thresholds and its metrics are unchanged, so the north star is reported at its session 11 value.

## User evidence

- **Raw artifact**: `pages/review.py`, `pages/admin.py`, `src/app/bulk.py`, `src/safety/`, `platform/services/issue-api/`, `docs/deployment_cost.md`

## Metrics snapshot
- **Autonomous routing rate @90% precision: 86.3%, unchanged** (served ensemble at 90.7% precision, full test set of 109,834 complaints, no team below the 85% floor). No model was retrained or rethresholded this session.
- **63 tests added across five files** (86 test files repo-wide, from 81): 25 on the safety gate, 16 on the issue service, 22 on bulk intake, highlighting and the admin data layer.

Operating cost of the system as built, measured in this repository and priced against vendor list rates (`docs/deployment_cost.md`):

| Measure | Value |
|---|---|
| Monthly cost, minimum viable deployment | $25–30 |
| Capacity of that deployment | 21.9M complaints/day |
| Serving-tier duty cycle at 1M complaints/year | 0.0127% |
| Annual cost at full US written-complaint volume | Under $1,000 |
| Marginal infrastructure cost per additional tenant | Approximately zero |
| Largest operating cost | LLM inference, ~10,000x the classifier it supports |
| Certification required to sell to a bank | SOC 2 Type II, $15–25k, 6–15 months |

- **The deployment decision turns on idle pricing rather than compute capacity.** The serving tier is unoccupied more than 99.98% of the time at realistic volume, and a 13.6s model load precludes scaling to zero.
- **Volume is not a constraint at any realistic scale.** The largest single recipient in the CFPB's 2025 data received 19,792 complaints for the year, which one CPU core clears in 78 seconds.

## What did not work
- **The safety gate has no caller.** Neither the application nor the platform router screens a complaint, so the gate does not yet affect the running product.
- **The misinformation judge fails open.** A timeout or unreadable reply allows the complaint through with `judge_error` recorded, which makes the check advisory in its current configuration.
- **Agentic token usage remains estimated.** The 2,900–4,100 tokens per complaint used in the cost analysis are derived from the prompt builders rather than metered from traffic, and the LLM layer is the largest operating cost.
- **Cloud and compliance figures are list prices.** They vary by region and provider and require confirmation against a quotation.

## Challenges / blockers
- **No continuous integration.** 63 tests were added this session by four contributors, none of which run on a pull request. Carried from session 11.
- **Certification, not engineering, constrains revenue.** The product has been operational since session 07 and remains unpurchasable by a regulated institution, as SOC 2 Type II is a procurement precondition requiring 6–15 months.
- **Improved Model 3 remains outside the ensemble**, on an unmerged branch.

## Next week's goal
  - Wire the safety gate into the application and the platform router, with a blocked-submission path that persists nothing.
  - Add a CI pipeline that runs the test suites on every pull request.
  - Meter the agentic layer's token usage to replace the estimated per-complaint cost.
  - Fold Improved Model 3 into the ensemble once fully trained.
  - Use the review queue's assignments as a first in-product evaluation set.

## Individual contributions
- **Sukriti Srivastava (Product)** — Added bulk complaint upload (up to 2,000 per file, with a CSV export of tracking IDs), an admin analytics page (`/admin`), and a human review queue (`/review`) where an admin assigns low-confidence complaints to a team, with 25 tests. Redesigned the frontend and fixed the explanation phrases that were never shown, now highlighted in the complaint text. Prepared the Supabase migration for review assignments.
- **Chaitanya Bagul (Engineering)** — Built the content safety gate (`src/safety/`), which screens a complaint before it reaches the router so abusive and fabricated submissions are refused rather than classified. Threats, harassment and profanity are matched deterministically, including disguised spellings, and claims about the outside world are assessed by an LLM judge on the agentic backend. A blocked complaint is not classified, routed or stored, and the submitter is given the reason. 25 tests. (PR #45)
- **Salman Farcy (Data and Evaluation)** — Costed the full deployment in `docs/deployment_cost.md`, comparing AWS, Azure and GCP for the phase-1 platform and establishing that the decision turns on idle pricing, as the serving tier is unoccupied over 99.98% of the time and a 13.6s model load precludes scaling to zero. Priced the minimum viable deployment at $25–30 a month against a measured capacity of 21.9M complaints a day, then sized a single-tenant bank deployment against 2025 CFPB volumes and a multi-tenant service, where marginal cost per tenant is approximately zero. Identified SOC 2 certification as the constraint on revenue. (PR #44)
- **Sriramm S S (Users and Research)** — **TODO (Sriramm): add this session's work.**

## Lean canvas changes (if any)
- **Channels:** the product has two surfaces rather than one, a consumer intake and an operator console (`/admin`, `/review`), and accepts complaints in bulk.
- **Unfair advantage:** restated from model quality to data position. Each review assignment records the model's prediction against the reviewer's, so a deployment accumulates labelled evaluation data in the operator's own complaint mix.
- **Cost structure:** infrastructure is settled and immaterial at $25–30 a month; the material costs are LLM inference and certification.
