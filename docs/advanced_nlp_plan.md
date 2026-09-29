# ClariCase: NLP Plan for Team + Issue Classification

**Goal:** from the complaint narrative alone, predict (1) the **team** (11 classes, the routing decision) and (2) the **issue** (73 product-specific labels, the triage detail inside the team).
**Data:** `data/processed/complaints_product_issues_2024_2025.csv`: 1,044,615 complaints, 2024-01 to 2025-12.
**Out of scope for now:** Jev and agentic flows. The LLM work is in Phase 4 (§7).

---

## 1. Decision: team first, then issue within the predicted team

### Evidence
I ran a quick probe on the existing baseline splits (19,799 train / 6,601 test, TF-IDF + LinearSVC; script: `src/experiments/team_vs_issue_probe.py`):

| Strategy | Team acc | Team macro-F1 | Issue acc | Issue macro-F1 |
|---|---:|---:|---:|---:|
| **A. Direct team classifier** | **0.778** | **0.780** | – | – |
| B. Flat 73-way issue classifier, team looked up from the issue | 0.741 | 0.741 | 0.496 | 0.338 |
| **C. Hierarchical: team, then issue model per team** | 0.778 | 0.780 | **0.521** | **0.361** |
| Ceiling: perfect team, then issue model | 1.000 | 1.000 | 0.641 | 0.470 |

What this shows:
1. **Team-first routes better.** Predicting the team directly beats getting it from a 73-way issue prediction by about 4 macro-F1 points. Routing is the product, so this settles it.
2. **Hierarchical beats flat for issue as well** (+2.3 macro-F1). Within one team the model only has to separate 4–12 issues, not 73.
3. **Issue is inherently noisy.** Even with the team given, issue accuracy is only 64%, and 46–52% for T09 and T11. Consumers pick issues from a dropdown, and several issue names overlap. So treat issue as a **ranked top-3 suggestion**, not a hard label.
4. **Team errors carry into issue errors** (0.470 ceiling → 0.361). The transformer design below reduces that with a shared encoder and soft decoding.

Caveat: this probe used the balanced 3k-per-team sample with a random split. The numbers will move on the real temporal split, but the ranking of strategies should hold.

### Final architecture: one encoder, two heads, hierarchical decoding

```
complaint ─► PII scrub ─► transformer encoder (shared) ─► pooled vector h
                                                  ├─► Team head   : p(team | x)             (11)
                                                  └─► Issue head  : p(issue | team, x)      (73, masked to the issues of each team)

Decode:  score(team, issue) = p(team|x) · p(issue|team,x)
         route  = argmax_team p(team|x)               ← team decision (calibrated, with abstain)
         issues = top-3 issues within routed team      ← triage suggestions
```

- **Training loss:** `L = CE_team + λ · CE_issue` (λ = 0.5 to start). The issue loss uses teacher forcing: it is computed within the *true* team during training.
- **Why a single model beats separate models:** the team head gets extra supervision from the fine-grained issue signal (multi-task regularization), and it is one model to train, serve, and calibrate.
- **Build order:** 1) team-only model → 2) add the issue head → 3) compare against a flat 73-way head as an ablation.

---

## 2. How to divide the data

### 2.1 Main split: temporal + leakage-group aware
| Split | Window | Records |
|---|---|---:|
| Train | 2024-01-01 → 2025-06-30 | 777,724 |
| Validation | 2025-07-01 → 2025-09-30 | 155,917 |
| Test (touch once) | 2025-10-01 → 2025-12-31 | 110,974 |

- **No `leakage_group_id` spans two splits.** Val/test records whose template group already appears in train are dropped, and the number dropped is logged.
- Val and test keep the **natural class mix**, which is what production looks like.
- Nine issues have fewer than 100 test records, so report per-issue **bootstrap 95% CIs**.

### 2.2 Transformer training set: capped per issue
- Credit Reporting is 62% of the data. **Cap each issue at 15,000 training records** (≈300k rows), sampled across months and companies. Issues below the cap keep every row.
- Compensate for the cap with class-weighted loss on the team head and label smoothing (0.1) on the issue head.
- TF-IDF and fastText baselines are cheap enough to train on the **full** 778k.

### 2.3 Label clean-up before issue training
- Merge CFPB wording variants: T01_I007 into T01_I008 ("existing issue" vs "existing problem"), and T01_I003 into T01_I001 (identity-theft monitoring). That gives 73 → 71 labels. Keep the merge table in `data/processed/issue_canonical.csv`.
- Keep product-specific issues separate even when they share a name (for example "Incorrect information on your report" under 6 teams). Masking by team handles them.

### 2.4 Extra evaluation sets
- **Balanced team test** (~600 per team), to compare with the NB baseline (0.732 macro-F1).
- **Hard-pair set:** Debt Collection ↔ Credit Reporting and Money Transfer ↔ Checking.
- **Unlabeled pool:** the 568k narratives excluded for conflicting labels are used **only** for domain pretraining (§3, step 6). Never as labels, never for evaluation.

---

## 3. Technique ladder: from TF-IDF to a fine-tuned transformer

Each step is one row in the results table, and each step should beat the one before it.

| # | Technique | Why it matters | Output |
|---|---|---|---|
| 1 | **TF-IDF (word 1–2 + char 3–5 grams) + LinearSVC / LogReg** | Strong sparse baseline; char n-grams catch typos and redaction noise | Baseline B |
| 2 | **fastText supervised** (subword embeddings) | Fast neural bag of n-grams; trains on all 778k in minutes | Model F |
| 3 | **Frozen sentence embeddings + LogReg** (bge-base / e5-base / gte) | "Transformer without fine-tuning" probe; shows what fine-tuning adds | Model E |
| 4 | **Fine-tuned BERT-base** (team head) | The classic BERT reference point | Model T1 |
| 5 | **Fine-tuned DeBERTa-v3-base and ModernBERT-base** | Stronger encoders; ModernBERT handles 1–8k tokens (14% of complaints exceed 384 words) | T2 / T3 |
| 6 | **Domain-adaptive pretraining (DAPT):** continue MLM on ≈1.3M complaint texts before fine-tuning | Learns FCRA/FDCPA/Zelle vocabulary, the root of the worst confusions | T4 |
| 7 | **Hierarchical multi-task heads** (§1) | Adds issue; regularizes team | T5 |
| 8 | **Calibration + selective routing** (temperature scaling, per-team thresholds, conformal sets) | Turns probabilities into "auto-route vs send to a human" | T6 |
| 9 | **Ensemble:** transformer + TF-IDF logit blend | Usually +0.5–1.5 F1 for little effort | T7 |

### Fine-tuning recipe (steps 4–7)
**Freeze → gradual unfreeze, with layer-wise learning-rate decay:**

| Stage | Trainable | LR | Epochs |
|---|---|---|---|
| 1 | heads only (encoder frozen: linear probe) | 1e-3 | 1 |
| 2 | top ⅓ of layers + heads | 5e-5 | 1 |
| 3 | all layers, LLRD = 0.85 per layer, cosine schedule, warmup 6% | 3e-5 | 1–2 |

- **Sequence length:** 512 (BERT/DeBERTa) or 1024 (ModernBERT), with **head+tail truncation** (first 75% + last 25% of tokens, because the actual ask is often at the end). Ablate against chunk-and-mean-pool.
- **Imbalance:** class-weighted CE or focal loss (γ = 2), chosen on validation.
- **Efficiency:** bf16, dynamic padding, length-bucketed batches, gradient checkpointing.
- **Ablation:** **LoRA (r = 16)** vs full fine-tuning. Report F1, GPU-hours, and trainable parameters.
- **Seeds:** 3 per final config; report mean ± std.

### Supporting NLP components
- **PII scrubbing** (`src/privacy/`): Presidio + spaCy NER + regex recognizers (account/card with Luhn check, SSN, phone, email, address). Normalize CFPB redactions (`XXXX` → `[REDACTED]`, `{$1,200}` → `[AMOUNT]`, `XX/XX/2024` → `[DATE]`) and add these as **special tokens**. Run the scrubber in both training and inference so the two see the same text. Keep company names, since they carry routing signal.
- **Label-noise audit:** cleanlab confident learning on 5-fold out-of-fold predictions. It estimates the label-error rate, which session 04 listed as "unmeasured". Optionally retrain without the flagged rows.
- **Explainability:** Integrated Gradients (Captum) on the team logit, shown as highlighted phrases in the UI.

---

## 4. Metrics

| Level | Metrics |
|---|---|
| Team (primary) | macro-F1, accuracy, per-team F1 with CI, confusion matrix |
| Issue | macro-F1, **top-3 accuracy**, per-issue F1 with CI |
| Joint | exact match (team **and** issue correct) |
| Routing (north star) | **auto-route coverage @ 95% precision** (currently 26.4%), ECE |
| Ops | latency (ms per complaint, CPU and GPU), model size |

---

## 5. Results table to fill in

| ID | Model | Team mF1 | Issue mF1 | Issue top-3 | Auto-route @95%P |
|---|---|---|---|---|---|
| B0 | Majority dummy | | | | |
| B1 | TF-IDF + NB (existing) | | | | |
| B2 | TF-IDF word+char + LinearSVC, hierarchical | | | | |
| F | fastText | | | | |
| E | Frozen bge/e5 + LogReg | | | | |
| T1 | BERT-base (team) | | | | |
| T2 | DeBERTa-v3-base (team) | | | | |
| T3 | ModernBERT-base (team) | | | | |
| T4 | best of T2/T3 + DAPT | | | | |
| T5 | T4 + hierarchical issue head | | | | |
| T6 | T5 + calibration + conformal | | | | |
| T7 | T6 + TF-IDF ensemble | | | | |
| L1 | T5 with LoRA | | | | |

---

## 6. Serving and UI

- **FastAPI** `/predict` → `{team, team_conf, route: auto | review, conformal_set, issues_top3, pii_entities, attributions}`
- **Streamlit "Triage Console"**, with four tabs:
  1. **Single:** paste text → scrubbed view → team card with an Auto-route or Review badge → top-3 issues → highlighted evidence words.
  2. **Batch:** upload a CSV → predictions and the auto/review queue split → download.
  3. **Compare:** TF-IDF vs BERT vs DeBERTa/ModernBERT on the same text.
  4. **Monitor:** confidence histogram, auto-route rate, team mix over time, latest confusion matrix.
- A **"Correct label"** button logs reviewer overrides to SQLite. That becomes future training data and the hook for the LLM phase.

---

## 7. Where an LLM fits (Phase 4, later)

**The main classifier is not an LLM.** We have about 1M labeled examples, and a fine-tuned encoder is typically more accurate on in-domain classification like this, 100–1000× cheaper per complaint, faster, and easier to calibrate. The LLM handles the parts an encoder does badly:

| Use | Where in the pipeline | Why an LLM |
|---|---|---|
| **1. Low-confidence adjudicator** | Only complaints the encoder sends to "review" (conformal set > 1 team, or confidence below threshold) | Reads the full text plus the 2–3 candidate teams/issues with their descriptions, and picks one with a rationale. The LLM sees a small share of the traffic, not all of it. |
| **2. Label-noise auditor** | Offline, on cleanlab-flagged training rows | Judges whether the consumer's issue label fits the text, giving a measured noise rate for the report. |
| **3. Issue taxonomy descriptions** | Offline, once | Writes a precise definition and inclusion/exclusion rules for each of the 71 issues. These are useful for the adjudicator prompt and for annotators. |
| **4. Data augmentation for thin issues** | Offline, train only | Paraphrases for issues with about 500 records (T11, T09, T10). Never used in val/test. |
| **5. Case summary for the assigned team** | After routing, in the UI | A two-line summary plus extracted fields (amount, dates, account type). |
| **6. Agentic flows** | Future | Tool use (look up the company, draft a response, check compliance), built on top of 1–5. |

Rules: only **PII-scrubbed** text goes to an LLM, outputs are constrained to the allowed label set (JSON schema), and every LLM decision is logged next to the encoder's.

---

## 8. Timeline (weekly sessions)

| Week | Deliverable |
|---|---|
| 1 | CSV → Parquet; `build_splits.py` (temporal + group-aware + cap); issue canonicalization; B1/B2 on the new split |
| 2 | PII scrubber + PII test set; fastText (F) and frozen-embedding (E) baselines |
| 3 | Fine-tune BERT / DeBERTa / ModernBERT team heads (T1–T3) with the freeze → unfreeze recipe; start DAPT in the background |
| 4 | DAPT model (T4) + hierarchical issue head (T5); cleanlab audit |
| 5 | Calibration, conformal routing, ensemble (T6–T7); LoRA ablation; error analysis on the hard pairs |
| 6 | FastAPI + Streamlit console deployed; final report |
| Later | Phase 4 LLM components (§7), then agentic |

## 9. Proposed code layout
```
src/
├── data/build_splits.py          # temporal + group split, cap, canonical issues → parquet
├── privacy/scrubber.py           # Presidio + regex + placeholder normalization
├── baselines/{tfidf,fasttext,embed_probe}.py
├── models/
│   ├── dapt.py                   # continued MLM pretraining
│   ├── hier_model.py             # encoder + team head + masked issue head
│   ├── train.py                  # freeze/unfreeze, LLRD, class weights, LoRA flag
│   └── calibrate.py              # temperature, thresholds, conformal
├── eval/{evaluate,noise_audit}.py
├── experiments/team_vs_issue_probe.py
└── serve/api.py
app/streamlit_app.py
```
