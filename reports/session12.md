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
- **Human review queue** (`pages/review.py`, PR #43 branch `week-12`): complaints routed below the confidence threshold now wait in a queue at `/review`, where an admin reads the complaint and assigns a team. The model's original choice is kept alongside the admin's (`model_team_id`, `model_team_name`, `reviewed_at`, `review_note`), so every override is recoverable as evaluation data. This closes session 11's "human review has no workflow" blocker.
- **Admin analytics page** (`pages/admin.py`, `src/app/admin_common.py`): incoming volume, routing mix (auto-routed / needs review / unknown), and team distribution, behind an admin password when complaints are stored in Supabase. Not linked from the consumer pages.
- **Bulk intake** (`src/app/bulk.py`): up to 2,000 complaints per uploaded file, with a CSV export of the tracking IDs issued. The intake app is no longer single-complaint only.
- **Content safety gate** (`src/safety/`, PR #45): screens a complaint before anything else touches it. Threats, harassment and profanity are caught by deterministic token and pattern checks with no network call, including disguised forms (`f*ck`, `sh1t`, `f u c k`); false claims about the outside world are judged by an LLM on the same backend as the agentic stage. A blocked complaint is not classified, routed or stored, and the consumer is shown the reason. 25 tests.
- **Product-conditioned issue backend** (`platform/services/issue-api/`, PRs #42, #43): the issue classifier becomes its own platform service, with a catalogue conditioned on the product so only issues valid for that product can be returned, plus a Dockerfile, contracts and API tests.
- **Explanation highlighting** (`src/app/highlight.py`): the explanation phrases produced in session 11 were computed but never displayed; they are now highlighted inside the complaint text itself.
- **Deployment and cost analysis** (`docs/deployment_cost.md`, PR #44).

## What changed since session 11
- **The product now has an operator side.** Until this week the system could route a complaint but not hand one over. With `/review` and `/admin` a below-threshold complaint has a destination and a person, which is the half of the "route the clear cases, review the rest" design that was missing.
- **A complaint can now be refused before routing.** The safety gate adds an outcome the pipeline did not have: not routed, not reviewed, not stored. This protects the corpus from abusive and fabricated submissions rather than classifying them.
- **Review decisions are now captured as labels.** Each assignment records the model's choice and the admin's, which gives the first source of in-product ground truth independent of the CFPB corpus.
- **No new modelling this week.** The served ensemble, its thresholds and its metrics are unchanged from session 11, so the north star is unchanged at 86.3%. The week's work was product, platform and commercial.

## User evidence

- **Raw artifact**: `pages/review.py`, `pages/admin.py`, `src/app/bulk.py`, `src/safety/`, `platform/services/issue-api/`, `docs/deployment_cost.md`

## Metrics snapshot
- **Autonomous routing rate @90% precision: 86.3%, unchanged** (served ensemble at 90.7% precision, full test set of 109,834 complaints, no team below the 85% floor). No model was retrained or rethresholded this week.
- **Tests: 63 added across five new files** (86 test files repo-wide, up from 81) — 25 on the safety gate, 16 on the issue service, 22 on bulk intake, highlighting and the admin data layer.

Operating cost of the system as built, measured in this repository and priced against vendor list rates (`docs/deployment_cost.md`):

| Measure | Value |
|---|---|
| Monthly cost, minimum viable deployment | $25–30 |
| Capacity of that deployment | 21.9M complaints/day |
| Serving-tier duty cycle at 1M complaints/year | 0.0127% |
| Annual cost at full US written-complaint volume | Under $1,000 |
| Marginal infrastructure cost per additional tenant | Approximately zero |
| Largest operating cost | LLM inference, ~10,000x the classifier it supports |
| Cost of certification required to sell to a bank | SOC 2 Type II, $15–25k, 6–15 months |

- **The deployment choice turns on idle pricing, not compute.** The serving tier is unoccupied more than 99.98% of the time at realistic volume, and the 13.6s model load rules out scaling to zero, so the provider with the cheapest idle vCPU rate wins even where its active rate is the highest.
- **Volume is not the constraint at any realistic scale.** The largest single bank in the CFPB's 2025 data received 19,792 complaints for the year; one CPU core clears that in 78 seconds.

## What did not work
- **The safety gate is not wired in.** It is a tested library with no caller: neither the Streamlit app nor the platform router screens a complaint yet, so no submission is actually blocked in the running product.
- **The misinformation judge fails open by default.** On a timeout or an unreadable reply the complaint is allowed through with a `judge_error` recorded. This is the right default for a consumer complaint service, but it means the check is advisory rather than enforced.
- **Agentic token usage is still estimated, not metered.** The cost analysis uses 2,900–4,100 tokens per complaint derived from the prompt builders, and the LLM layer is the system's largest operating cost, so this is the figure most worth measuring directly.
- **Cloud and compliance figures are list prices.** They vary by region, provider and negotiation and should be confirmed against a quotation before they are relied on.

## Challenges / blockers
- **Still no continuous integration.** Four contributors added 63 tests this week and nothing runs them on a pull request. This was session 11's blocker and remains open.
- **The commercial blocker is certification, not engineering.** The product has been operational since session 07 and could not be purchased by a regulated institution today, because SOC 2 Type II is a procurement precondition and takes 6–15 months.
- **Improved Model 3 is still outside the ensemble** and its branch is unmerged.

## Next week's goal
  - Wire the safety gate into the Streamlit intake and the platform router, with a blocked-submission path that stores nothing.
  - Add a CI pipeline that runs the test suites on every pull request.
  - Meter the agentic layer's live token usage to replace the estimated per-complaint cost.
  - Fold Improved Model 3 into the ensemble once fully trained.
  - Use the review queue's assignments as a first in-product evaluation set.

## Individual contributions
- **Sukriti Srivastava (Product)** — Added bulk complaint upload (up to 2,000 per file, with a CSV export of tracking IDs), an admin analytics page (`/admin`), and a human review queue (`/review`) where an admin assigns low-confidence complaints to a team, with 25 tests. Redesigned the frontend and fixed the explanation phrases that were never shown, now highlighted in the complaint text. Prepared the Supabase migration for review assignments.
- **Chaitanya Bagul (Engineering)** — Built the content safety gate (`src/safety/`), which screens a complaint before it reaches the router so abusive and fabricated submissions are refused rather than classified. Threats, harassment and profanity are caught deterministically, including disguised spellings, and false claims about the outside world go to an LLM judge on the agentic backend; a blocked complaint is not classified, routed or stored, and the consumer is told why. 25 tests. (PR #45)
- **Salman Farcy (Data and Evaluation)** — Costed the full deployment in `docs/deployment_cost.md`: compared AWS, Azure and GCP for the phase-1 platform and found the decision turns on idle pricing rather than compute, since the serving tier is idle over 99.98% of the time and the 13.6s model load precludes scaling to zero. Priced the MVP at $25–30 a month against a measured capacity of 21.9M complaints a day, then sized a single-tenant bank deployment against real 2025 CFPB volumes and a multi-tenant service, where marginal cost per tenant is approximately zero. Concluded that SOC 2 certification, not engineering or cost, is the constraint on revenue. (PR #44)
- **Sriramm S S (Users and Research)** — **TODO (Sriramm): add this session's work.**

## Lean canvas changes (if any)
- **Channels:** the product now has two surfaces rather than one — a consumer intake and an operator console (`/admin`, `/review`) — and accepts complaints in bulk, which is how an institution would actually submit a backlog.
- **Unfair advantage:** reframed from model quality to data position. Every review assignment records the model's choice against the reviewer's, so the deployment generates its own labelled evaluation data in the operator's own complaint mix.
- **Cost structure:** infrastructure is settled and immaterial ($25–30 a month at four orders of magnitude more capacity than needed); the costs that matter are LLM inference and certification.
