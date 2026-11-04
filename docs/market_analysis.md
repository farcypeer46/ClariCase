# Market position and value case

Where ClariCase sits in the complaint-handling software market, who it competes
with, and what the routing layer is worth. Compiled November 2026.

**On the sources.** Market sizing comes from published summaries of paid
analyst reports, and the competitive and cost figures largely from vendor
blogs and comparison guides written by companies selling in this category.
Those have a commercial interest in the numbers they publish. Treat everything
below as directional. Figures drawn from this project's own artifacts are
marked as such and are reproducible from the repository.

---

## 1. Market size

| Source | Estimate |
|---|---|
| Coherent Market Insights | $3.11B (2026) → $5.83B (2033) |
| IMARC | $2.9B (2025) → $6.3B (2034), CAGR 8.57% |
| OMR Global | → $12.6B (2035) |

Complaint management software is a **$3B category growing at roughly 8–9% a
year**. The outlier high estimate (OMR) is a different segment definition
rather than a disagreement about the same market.

No source isolates the financial-services share. The qualitative signal is
consistent across them: regulated industries — financial services, healthcare,
telecoms — are the segment with the strongest compliance pressure to track,
document and report complaints.

## 2. Addressable volume

From this project's own corpus rather than a vendor estimate:

| Measure | Value |
|---|---|
| US financial complaints filed to the CFPB, 2024–25 | 8,177,233 |
| Per year | ~4.1M |
| Of those, carrying a written narrative | ~1M a year |
| Companies receiving them | 3,801 |

That is only complaints escalated to the federal regulator. Complaints
resolved inside a company never reach the CFPB, so the real triage volume is
larger by an unknown multiple.

## 3. Competitive landscape

| Vendor | Position |
|---|---|
| **Zendesk** | Native NLP ticket classification; acquired **Forethought** in March 2026 |
| **Forethought** (now Zendesk) | "Triage" classification and routing engine; 85–93% accuracy, vendor-reported |
| **Salesforce** | Service Cloud; Forethought and Fini both integrate with it |
| **Ada** | SOC 2 Type II, GDPR, HIPAA — the broadest published compliance footprint |
| **Fini** | Zendesk/Salesforce-coupled triage |

**The incumbents report accuracy alone.** None of the published figures states
a coverage rate at a precision floor. "85–93% accurate" does not say how many
tickets the system was willing to decide on, which is the number that
determines how much work actually leaves the queue.

This project measures the opposite way round: **61.5% of complaints routed
with no human at 94.8% precision**, on 109,834 held-out complaints, with the
confidence threshold fitted on validation and frozen before test. The two
figures are not comparable, and that is the point — a precision-constrained
coverage metric is a stricter claim than accuracy, and it is the one an
operations team can act on.

**Second difference: the training data.** Forethought learns from a customer's
own historical tickets, so its accuracy depends on that customer's label
quality. ClariCase is trained on 1,044,615 complaints across 3,801 companies,
so a new deployment does not start cold — and the label-quality problem was
measured rather than assumed (568,050 complaints with identical text under
conflicting labels were excluded during corpus construction).

## 4. Cost baseline

| Channel | Cost per contact |
|---|---|
| Self-service | $1.84 |
| Assisted (phone, chat, email) | $13.50 |
| North American mid-market/enterprise | $13–18 |
| Blended global average | $7–14 |
| US support representative | $19.74/hour |

**No published source isolates the cost of the triage step** from full
resolution. We decline to put a dollar figure on triage without measuring it;
doing so would be the kind of unfounded claim this project has avoided
elsewhere.

What can be stated from measured results: at 61.5% coverage a person makes
**2.6× fewer triage decisions**. Against ~4.1M US financial complaints a year,
that is roughly 2.5M routing decisions a year that need no human — before
counting complaints resolved inside companies and never escalated.

## 5. Where this leaves the product

**The defensible claim.** Not "more accurate than Zendesk" — we have not
benchmarked against it, and could not without their test set. The claim is
that ClariCase states coverage at a precision floor and declines to guess
below it, which the published alternatives do not.

**The weak spot.** A category growing at 8–9% with Zendesk and Salesforce
already in it is not an open field. Distribution, not accuracy, is the barrier:
incumbents own the ticketing system the complaint already lives in.

**Where the roadmap matters.** Routing is a feature of existing platforms.
Resolution — drafting the steps, acting on the straightforward cases,
escalating the rest — is not yet commodity, and it is where this project's
confidence-threshold discipline transfers directly.

---

## Sources

- [Coherent Market Insights — complaint management software](https://www.coherentmarketinsights.com/ja/洞察力/苦情管理ソフトウェア市場-8943/目次)
- [IMARC — global complaint management software market](https://www.imarcgroup.com/global-complaint-management-software-market)
- [OMR Global — complaint management software market share](https://www.omrglobal.com/press-release/global-complaint-management-software-market-share)
- [eesel.ai — cost per ticket benchmarks](https://www.eesel.ai/blog/cost-per-ticket-benchmarks)
- [Stealth Agents — customer support cost per ticket benchmarks 2026](https://stealthagents.com/research/customer-support-cost-per-ticket-benchmarks-2026)
- [usefini.com — ticket triage AI for Zendesk](https://www.usefini.com/guides/ticket-triage-ai-for-zendesk)
- [eesel.ai — Zendesk AI ticket classification](https://www.eesel.ai/blog/zendesk-ai-ticket-classification-complete-rundown-in-2026)
- [usefini.com — AI triage engines and regulatory SLAs](https://www.usefini.com/guides/ai-triage-engines-regulatory-complaints-48-hour-sla)
