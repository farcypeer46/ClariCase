# Implementation Plan: Improved Models (for Claude Code)

> **Audience:** Claude Code, working in this repository (`F:\NLP_MSML641`, Windows / PowerShell, Python venv at `.venv`).
> **Goal:** build a sequence of improved complaint classifiers that predict **team** (11 classes) and **issue** (product-specific issue, ~71 classes) from `complaint_text`. Each improvement lives in its **own folder** (`src/improved_model_N/`).
> **Current state:** `src/baseline_model_1/` = TF-IDF + Multinomial Naive Bayes, team only. Test macro-F1 0.732 on the balanced 3k/team group-random split (`data/processed/splits/`).
> Background and reasoning: `docs/advanced_nlp_plan.md`. This file is the step-by-step execution spec.

---

## 0. Ground rules (read before writing code)

1. **Never modify `src/baseline_model_1/`.** It is the frozen reference.
2. **One folder per improvement:** `src/improved_model_1/`, `src/improved_model_2/`, … Reusable shared code (data splits, label utilities, text normalization, metrics, device selection, results registry) goes in **`src/utils/`** so model folders stay small and import from it (`from src.utils.metrics import evaluate`). The empty folders `src/improved_model_2/` … `src/improved_model_5/` already exist as placeholders. Leave them empty until their section is started, and add an `__init__.py` only then.
3. **Model inputs:** `complaint_text` only. Never feed `product`, `sub_product`, `issue`, `sub_issue`, `team_*`, `product_issue_*`, IDs, group hashes, `company` or `source_*` columns to a model.
4. **Fit on train only.** Vectorizers, embedding models, class weights, scalers and calibration are all fit on train (calibration on validation). The test set is used once, for the final numbers.
5. **Leakage:** a `leakage_group_id` must never span two splits.
6. **Reproducibility:** `RANDOM_SEED = 42` everywhere; log library versions to the artifacts.
7. **Git:** work on a branch per model (`improved_model_1`, `improved_model_2`, …). **Never commit data or model binaries.** Add `data/processed/splits_v2/`, `*.joblib`, `*.npy`, `*.kv` and `src/**/artifacts/*.joblib` to `.gitignore`. Commit metrics JSON/CSV and docs.
8. **Memory:** the full CSV is 1.9 GB. Always read it with `usecols=[...]`, `dtype={"complaint_id": "string"}`, `keep_default_na=False` and `chunksize=100_000`. Convert it to Parquet once (step 1.1) and read the Parquet file afterwards.
9. **Housekeeping:** delete `data/processed/splits/w2v_corpus.csv`, a partial scratch file from an exploratory run. It is not needed.
10. **GPU available:** use it for every neural step (sentence embeddings, BERT fine-tuning, DAPT). Always select the device via `src/utils/device.py` (§0.1), never hard-code `"cuda"`, so the code still runs on a CPU-only teammate's laptop.

### 0.1 GPU setup (one-time, Windows)
1. Check the driver and CUDA version: `nvidia-smi`.
2. Install the PyTorch build that matches the driver (pick the command for your CUDA version at pytorch.org). For example:
   ```powershell
   python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```
   Do **not** put a CUDA-specific torch line in `requirements.txt`. List `torch` there and document the install command in the README, because teammates may have different GPUs or none.
3. Verify:
   ```powershell
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
4. `src/utils/device.py`:
   ```python
   def get_device(prefer: str = "auto") -> str        # "cuda" if available (and prefer != "cpu") else "cpu"
   def gpu_info() -> dict                             # name, total VRAM GB, torch/CUDA versions → saved in run_info.json
   def amp_dtype() -> torch.dtype | None              # bf16 if torch.cuda.is_bf16_supported() else fp16 on GPU; None on CPU
   def suggested_batch_size(vram_gb, task) -> int     # simple table below
   ```
   | Task (max_len) | ≤8 GB VRAM | 12–16 GB | ≥24 GB |
   |---|---|---|---|
   | Sentence encoding, base model (512) | 64 | 128 | 256 |
   | Fine-tune BERT/DeBERTa-base (512) | 8 + grad-accum 4 | 16 + grad-accum 2 | 32 |
   | Fine-tune ModernBERT-base (1024) | 4 + grad-accum 8 | 8 + grad-accum 4 | 16 + grad-accum 2 |
   Enable gradient checkpointing if batch size 8 still runs out of memory.
5. Windows notes: use `num_workers=0` in DataLoaders (or wrap entry points in `if __name__ == "__main__":`), and set `TOKENIZERS_PARALLELISM=false` to avoid warnings.
6. The scikit-learn models (NB, LinearSVC, LogisticRegression) run on **CPU**. That is fine, since they train in minutes. Do not add cuML (it is not supported on native Windows).

---

## 1. Shared foundation: `src/utils/`

```
src/utils/
├── __init__.py
├── config.py          # paths, seed, split dates, caps, column names
├── labels.py          # team/issue mappings, issue canonicalization, team→issues mask
├── build_splits.py    # CSV → Parquet, temporal + group-aware splits v2
├── data.py            # load_split(name) helpers
├── text.py            # normalize_text() (redaction/amount tokens), word tokenizer — reused by every model
├── device.py          # get_device(), gpu_info(), amp_dtype(), suggested_batch_size() (§0.1)
├── metrics.py         # all evaluation metrics (team, issue, joint, routing, ECE)
└── registry.py        # append results to reports/model_comparison.csv
```

Target repository layout after this plan:
```
src/
├── baseline_model_1/     # existing, untouched
├── experiments/          # existing probe script
├── utils/                # shared, reusable (this section)
├── improved_model_1/     # TF-IDF + LinearSVC, hierarchical (§2)
├── improved_model_2/     # placeholder (empty) → embeddings (§3)
├── improved_model_3/     # placeholder (empty) → fine-tuned BERT-base (§4)
├── improved_model_4/     # placeholder (empty) → DeBERTa/ModernBERT hierarchical multi-task (§4)
└── improved_model_5/     # placeholder (empty) → DAPT + calibration/conformal + ensemble (§4)
```

### 1.1 `config.py`
```python
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FULL_CSV = PROCESSED / "complaints_product_issues_2024_2025.csv"
FULL_PARQUET = PROCESSED / "complaints_product_issues_2024_2025.parquet"
LEGACY_SPLITS = PROCESSED / "splits"          # baseline 3k/team split (read-only)
SPLITS_V2 = PROCESSED / "splits_v2"
ISSUE_MAPPING = PROCESSED / "issue_mapping.csv"
TEAM_MAPPING = PROCESSED / "team_mapping.csv"
REPORTS = ROOT / "reports"

TEXT_COL, TEAM_COL, ISSUE_COL = "complaint_text", "team_id", "issue_id"   # issue_id = canonical product_issue_id
GROUP_COL, DATE_COL, ID_COL = "leakage_group_id", "date_received", "complaint_id"
USECOLS = [ID_COL, DATE_COL, "company", TEXT_COL, TEAM_COL, "product_issue_id", GROUP_COL]

RANDOM_SEED = 42
TRAIN_END = "2025-06-30"; VAL_END = "2025-09-30"; TEST_END = "2025-12-31"
TRAIN_CAP_PER_ISSUE = 10_000      # ≈259k train rows; set None to use all 778k
BALANCED_TEST_PER_TEAM = 600
MERGE_ISSUE_VARIANTS = True
ISSUE_VARIANT_MERGES = {"T01_I007": "T01_I008",   # "...existing issue" → "...existing problem"
                        "T01_I003": "T01_I001"}   # identity-theft monitoring wording variants
```

### 1.2 `labels.py`
- `load_issue_mapping()` returns the DataFrame from `issue_mapping.csv`.
- `canonical_issue(pid)` applies `ISSUE_VARIANT_MERGES` when `MERGE_ISSUE_VARIANTS` is on.
- Write `data/processed/issue_canonical.csv` (`product_issue_id, issue_id, team_id, issue_label`).
- `issue_to_team: dict[str, str]` and `team_to_issues: dict[str, list[str]]`.
- `team_issue_mask(issue_classes, team_classes) -> np.ndarray[bool]` with shape (n_teams, n_issues).

### 1.3 `build_splits.py` (run once: `python -m src.utils.build_splits`)
1. If `FULL_PARQUET` is missing, stream the CSV in chunks with `USECOLS` and write Parquet (pyarrow).
2. Add `issue_id = canonical_issue(product_issue_id)`.
3. **Temporal assignment:** train ≤ 2025-06-30 < val ≤ 2025-09-30 < test ≤ 2025-12-31.
4. **Group rule:** each `leakage_group_id` belongs to the split of its **earliest** record. Drop that group's records that fall in a later split, and log how many per split.
5. **Train cap:** for each `issue_id` in train, keep at most `TRAIN_CAP_PER_ISSUE` rows (`groupby("issue_id").sample(..., random_state=RANDOM_SEED)`, or all rows when below the cap). Save the **uncapped** train IDs as well (`train_full_ids.parquet`); model 2 uses them for unsupervised embedding training.
6. Val and test keep their **natural distribution** (no capping).
7. `test_balanced`: sample ≤ `BALANCED_TEST_PER_TEAM` per team from test. This set is comparable to the baseline's balanced evaluation.
8. Write `SPLITS_V2/{train,val,test,test_balanced}.parquet` with columns `complaint_id, date_received, company, complaint_text, team_id, issue_id, leakage_group_id`.
9. Write `SPLITS_V2/split_report.json`: rows per split, per team and per issue; dropped cross-boundary group rows; date ranges; and assertions that group sets are disjoint and every split has all 11 teams.

Expected sizes before group drops: train ≈259k (capped) / 778k (full), val ≈156k, test ≈111k.

### 1.4 `metrics.py`
Implement `evaluate(y_team, p_team_scores, team_classes, y_issue=None, p_issue_scores=None, issue_classes=None) -> dict`. It should return:

| Group | Metrics |
|---|---|
| Team | accuracy, macro-F1, weighted-F1, per-team P/R/F1/support, confusion matrix (11×11) |
| Issue | accuracy, macro-F1, weighted-F1, **top-3 accuracy**, per-issue P/R/F1/support |
| Joint | exact match (team **and** issue correct) |
| Oracle | issue accuracy / macro-F1 when the **true** team is given (ceiling for the hierarchy) |
| Routing | when calibrated probabilities are available: **coverage @ 90% and @ 95% precision** (sort by max prob descending; largest covered fraction with precision ≥ target), **ECE** (15 equal-width bins) |
| Uncertainty | bootstrap 95% CI (1,000 resamples) for team macro-F1 and issue macro-F1 |
| Hard pairs | recall of T02 (Debt Collection) and share of it predicted as T01 (Credit Reporting); recall of T05 (Money Transfer) and share predicted as T04 (Checking) |

`save_eval(result, out_dir, prefix)` writes `metrics_{prefix}.json`, `classification_report_team_{prefix}.csv`, `classification_report_issue_{prefix}.csv` and `confusion_matrix_team_{prefix}.csv`.

### 1.5 `registry.py`
`log_result(model_name, split_name, metrics_dict)` appends one row to `reports/model_comparison.csv` with these columns: `timestamp, model, split, team_acc, team_macro_f1, issue_macro_f1, issue_top3, joint_exact, coverage_95p, ece, n_test, git_commit`. Every model folder calls it, so the comparison table grows with each improvement.

---

## 2. `src/improved_model_1/`: TF-IDF + Linear SVM, hierarchical team → issue

### 2.1 Why this design
A probe on the legacy split (`src/experiments/team_vs_issue_probe.py`) gave:

| Strategy | Team macro-F1 | Issue macro-F1 |
|---|---:|---:|
| Direct team classifier | **0.780** | – |
| Flat 73-way issue, team derived via mapping | 0.741 | 0.338 |
| **Hierarchical: team, then per-team issue model** | **0.780** | **0.361** |
| Ceiling: perfect team → issue | 1.000 | 0.470 |

So the model **predicts the team first**, then predicts the **issue within the predicted team**, and reports issue as a **top-3 suggestion**.

### 2.2 Folder layout
```
src/improved_model_1/
├── __init__.py
├── config.py        # model-specific hyperparameters and grids
├── features.py      # build_vectorizer()
├── model.py         # HierarchicalSVM class
├── train.py         # entry point: tune → fit → calibrate → evaluate → save
├── predict.py       # CLI inference, JSON output
└── artifacts/       # generated
```

### 2.3 `features.py`
```python
def build_vectorizer(use_char=True):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, max_features=200_000,
                           sublinear_tf=True, strip_accents="unicode", dtype=np.float32,
                           preprocessor=normalize_text)
    if not use_char:
        return word
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=300_000,
                           sublinear_tf=True, dtype=np.float32, preprocessor=normalize_text)
    return FeatureUnion([("word", word), ("char", char)])
```
`normalize_text`: lowercase, `{$1,234.00}` → ` amounttok `, runs of `XX…` (including `XX/XX/XXXX`) → ` redacttok `, collapse whitespace. It lives in `src/utils/text.py` so later models reuse it.

If memory is tight on 259k rows, set `use_char=False` or `TRAIN_CAP_PER_ISSUE=5000` and note it in the doc.

### 2.4 `model.py`: `HierarchicalSVM`
- `team_clf = LinearSVC(C=C_team, class_weight="balanced", dual="auto", max_iter=5000)`
- `issue_clfs = {team: LinearSVC(C=C_issue, class_weight="balanced", ...)}`: one per team, trained only on that team's rows. A team with a single issue gets a constant predictor.
- `flat_issue_clf`: an **ablation only**, a single LinearSVC over all issues.
- `fit(X, y_team, y_issue)`
- `decision_team(X)` → (n, 11) scores. `predict_team(X)`.
- `predict_issue_topk(X, teams, k=3)`: for each row, use the issue classifier of the given team (predicted at inference, true team for the oracle metric) and return the top-k issue IDs with scores.
- **Calibration** (LinearSVC has no probabilities): wrap the fitted team classifier with `CalibratedClassifierCV(FrozenEstimator(team_clf), method="sigmoid")` (sklearn ≥ 1.6), and fit the calibrator on **validation** features. Do the same for each per-team issue classifier where the validation set has ≥ 2 classes; otherwise fall back to softmax over decision scores. Save the calibrated models.

### 2.5 `train.py` (run: `python -m src.improved_model_1.train [--split v2|legacy] [--no-char] [--quiet]`)
1. Load `train`, `val` and `test` from `splits_v2` (default) or the legacy `splits/` (for the like-for-like comparison with NB; legacy has no `issue_id`, so derive it from `(product, issue)` via `issue_mapping.csv`, as in the probe script).
2. Fit the vectorizer on train. Transform val and test.
3. **Tune on val** (grid in `config.py`): `C_team ∈ {0.1, 0.25, 0.5, 1.0}` by team macro-F1, then `C_issue ∈ {0.1, 0.25, 0.5, 1.0}` by val issue macro-F1 under the oracle team. Log the grid to `artifacts/tuning.json`.
4. Refit on train with the best C values. Do not merge train and val; val stays the calibration set.
5. Calibrate on val (2.4).
6. Evaluate on **test** and **test_balanced** (v2) or test (legacy) with `utils.metrics.evaluate`, for:
   - `improved_model_1` (hierarchical, calibrated)
   - `flat_issue_ablation` (flat LinearSVC → team via mapping)
   - `nb_reference`: retrain the **baseline recipe** (TF-IDF 1–2 + MultinomialNB α=0.1, balanced sample weights) on the **same** split, so every split has an NB row. Do not import from `baseline_model_1`; reimplement it in 10 lines to keep folders independent.
   - `dummy_most_frequent`
7. Save to `artifacts/`: `model.joblib` (vectorizer + HierarchicalSVM + calibrators), all metric files, `predictions_test.csv` (`complaint_id, true_team, pred_team, team_conf, true_issue, issue_top1, issue_top2, issue_top3`), `run_info.json` (versions, split, C values, n rows, runtime).
8. Call `registry.log_result` for each evaluated model and split.

### 2.6 `predict.py`
```
python -m src.improved_model_1.predict "Debt collector keeps calling about a loan I already paid"
```
Returns:
```json
{"team_id": "T02", "team_name": "Debt Collection Team", "team_confidence": 0.91,
 "team_top3": [["T02", 0.91], ["T01", 0.06], ["T04", 0.01]],
 "issue_top3": [["T02_I001", "Attempts to collect debt not owed", 0.64], ...],
 "route": "auto" | "review"}
```
`route = "auto"` when `team_confidence ≥` the per-run threshold that achieved 95% precision on val (saved in `run_info.json`).

### 2.7 Documentation
Write `docs/improved_model_1.md` in the same style as `docs/baseline_model_2.md`, covering objective, design, how to run, the results table (NB vs SVM, legacy and v2 splits), the hard-pair analysis, and what did not work.

### 2.8 Acceptance criteria
- [ ] `build_splits` passes all assertions, and `split_report.json` exists.
- [ ] On the **legacy** split, team macro-F1 of improved_model_1 **>** NB reference (0.732). The probe suggests ≈0.78.
- [ ] Hierarchical issue macro-F1 **≥** flat ablation on both splits.
- [ ] `reports/model_comparison.csv` contains rows for dummy, NB, SVM and flat-ablation.
- [ ] `predict.py` works from a fresh shell using only `artifacts/`.

---

## 3. `src/improved_model_2/`: better text representations (embeddings)

Same hierarchy, labels, splits, metrics and registry as model 1. **Only the features change.** Each variant below is one config, logged separately. The best variant on **val** becomes the official `improved_model_2`.

### 3.1 Folder layout
```
src/improved_model_2/
├── config.py
├── embedders.py       # Word2VecEmbedder, FastTextEmbedder, SentenceEmbedder, cache helpers
├── train_word_vectors.py   # trains Word2Vec / FastText on train-period text only
├── encode_sentences.py     # caches sentence-transformer embeddings to .npy
├── train.py           # --variant {w2v_mean, w2v_tfidf, ft_tfidf, sbert, hybrid_tfidf_w2v, hybrid_tfidf_sbert}
├── predict.py
└── artifacts/
```
New requirements (add to `requirements.txt`): `gensim`, `sentence-transformers`, `torch`. Install the CUDA build of torch as described in §0.1.

### 3.2 Variants

| Variant | Representation | Classifier |
|---|---|---|
| `w2v_mean` | Word2Vec (CBOW, 300d, window 8, min_count 5, 5 epochs) trained on **uncapped train-period** texts (`train_full_ids`, never val/test); mean of word vectors | StandardScaler → LogisticRegression (balanced, max_iter 3000) |
| `w2v_tfidf` | Same vectors, **IDF-weighted** mean (IDF from train) | same |
| `ft_tfidf` | gensim **FastText** (subword n-grams 3–6, 300d), IDF-weighted mean; handles typos like "dipute" and "zell" | same |
| `sbert` | Frozen sentence embeddings: `BAAI/bge-base-en-v1.5` (default) and `intfloat/e5-base-v2` (prefix `"query: "`), `max_seq_length=512`, `normalize_embeddings=True` | LogisticRegression |
| `hybrid_tfidf_w2v` | `hstack([TF-IDF from model 1, α · w2v_tfidf])`; tune α ∈ {0.05, 0.1, 0.2, 0.4} on val | LinearSVC (as model 1) |
| `hybrid_tfidf_sbert` | `hstack([TF-IDF, α · sbert])`, with α tuned the same way | LinearSVC |

Tokenization for word vectors: the same `normalize_text`, then regex `[a-z]+(?:'[a-z]+)?`. Keep `redacttok` and `amounttok` as tokens.

### 3.3 Practical notes
- **Run sentence encoding on the GPU.** `encode_sentences.py` takes `--device {auto,cuda,cpu}` (default `auto` via `utils.device.get_device()`), `--model`, `--batch-size` (default from `suggested_batch_size`), `--max-len 512` and `--limit`. Load the model with `model.half()` (or `model_kwargs={"torch_dtype": amp_dtype()}`) on GPU, and wrap encoding in `torch.inference_mode()`. Cache to `artifacts/emb_{model}_{split}.npy` (float16 to save disk) so each split is encoded once.
- On GPU, encode **all** splits: capped train ≈259k, val ≈156k, test ≈111k. Expect minutes to tens of minutes per model, depending on the card. CPU fallback only if CUDA is unavailable: use `BAAI/bge-small-en-v1.5` at 256 tokens and state that in the doc.
- Sort texts by length before batching (restore the original order afterwards). With GPU this is the biggest speed-up.
- Word2Vec and FastText (gensim) are CPU-only; use `workers = os.cpu_count() - 1`. LogisticRegression and LinearSVC on the embeddings stay on CPU.
- Log `gpu_info()` and the encoding throughput (docs/s) into `run_info.json`.
- Expectation to write in the doc: averaged word vectors often **do not beat** TF-IDF on long documents. The expected gains come from the **hybrid** variants and frozen sentence embeddings. Report whatever happens.
- Save the Word2Vec nearest neighbours for 10 domain words (`repossession, zelle, fdcpa, collector, dispute, chargeback, escrow, overdraft, garnishment, forbearance`) to `artifacts/w2v_neighbors.json` for the report.

### 3.4 Acceptance criteria
- [ ] All 6 variants logged in `reports/model_comparison.csv` (val and test).
- [ ] `docs/improved_model_2.md` with a table comparing NB → model 1 → each model-2 variant, and the chosen variant justified **by validation**, not test.

---

## 4. Roadmap for later folders (spec to be expanded when started)

| Folder | Improvement |
|---|---|
| `src/improved_model_3/` | **Fine-tuned BERT-base**, team head only. Hugging Face `Trainer`; freeze → gradual unfreeze (heads → top ⅓ layers → all with layer-wise LR decay 0.85); max_len 512 with head+tail truncation; class-weighted CE; bf16 on GPU. |
| `src/improved_model_4/` | **DeBERTa-v3-base / ModernBERT-base** with **hierarchical multi-task heads** (team + masked issue, loss = CE_team + 0.5·CE_issue); special tokens `[REDACTED]`, `[AMOUNT]`, `[DATE]`. |
| `src/improved_model_5/` | **Domain-adaptive pretraining** (continued MLM on train-period complaint text) before model 4's fine-tuning. |
| `src/improved_model_6/` | **Calibration + selective routing:** temperature scaling, per-team thresholds, conformal prediction sets; ensemble with the model-1 TF-IDF SVM. |
| later | PII scrubbing layer (Presidio) in `src/utils/`; LLM adjudicator for the low-confidence "review" queue; UI (FastAPI + Streamlit). |

Every folder follows the same contract: `train.py`, `predict.py`, `artifacts/`, `docs/improved_model_N.md`, and results appended to `reports/model_comparison.csv`.

---

## 5. Execution order for Claude Code

1. Create branch `improved_model_1` from `baseline_model_2`.
2. Delete `data/processed/splits/w2v_corpus.csv`; update `.gitignore` (rule 0.7).
3. Implement `src/utils/` (§1) and run `python -m src.utils.build_splits`. Check `split_report.json`.
4. Implement `src/improved_model_1/` (§2). Run `python -m src.improved_model_1.train --split legacy`, then `--split v2`.
5. Write `docs/improved_model_1.md`. Commit code, metrics and docs (no data or binaries).
6. Create branch `improved_model_2` and implement §3: run `train_word_vectors`, then `encode_sentences` (GPU if available), then `train --variant ...` for each variant.
7. Write `docs/improved_model_2.md`. Commit.
8. Stop and report the comparison table to the user before starting model 3.
