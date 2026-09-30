# Implementation Plan: Improved Models (for Claude Code)

> **Audience:** Claude Code, working in this repository (`F:\NLP_MSML641`, Windows / PowerShell, Python 3.14 venv at `.venv`).
> **Goal:** build a sequence of improved complaint classifiers that predict **team** (11 classes) and **issue** (product-specific issue, ~71 classes) from `complaint_text`. Each improvement lives in its **own folder** (`src/improved_model_N/`). Then plug the winner into the existing Streamlit app without breaking it.
> **Model to beat: `src/baseline_model_2/`**, the model running in the app. TF-IDF + Multinomial NB with isotonic calibration, team only. Numbers are on the temporal test set (Oct–Dec 2025, 109,834 complaints at the real team mix):
>
> | Accuracy | Macro-F1 | North star (auto-routed @ 95% precision) | Test precision at that threshold | ECE | Guardrail |
> |---:|---:|---:|---:|---:|---|
> | 0.800 | 0.647 | **46.5%** (threshold 0.872, fit on validation) | 94.4% | 0.026 | T09 below the 0.85 floor |
>
> Background and reasoning: `docs/advanced_nlp_plan.md`. This file is the step-by-step execution spec.
> **Revision note (after pulling latest `main`):** the plan now reuses baseline_model_2's split protocol, evaluation harness and text cleaning, writes artifacts to `docs/metrics/improved_model_N/` like baseline 2, keeps `requirements.txt` deploy-safe, and adds an app-integration phase (§5). The old 3k-per-team group-random split is retired.

---

## 0. Ground rules (read before writing code)

1. **Do not modify** any of these:
   - `src/baseline_model_1/`
   - `src/baseline_model_2/`
   - `src/data/`
   - `data/preparation/`
   - `data/processed/splits/` (owned by baseline_model_2)
   - `docs/metrics/baseline_model_1/`, `docs/metrics/baseline_model_2/`
   - the two LFS zips in `data/`

   **Do not run `python -m src.baseline_model_2.train`.** It would rebuild `data/processed/splits/` and overwrite the committed baseline artifacts that the live app loads. Importing *read-only* helpers from `src.baseline_model_2` is allowed but discouraged; copy what is needed into `src/utils/` (§1.6) and prove parity instead.
2. **One folder per improvement:** `src/improved_model_1/`, `src/improved_model_2/`, … Reusable shared code (data splits, label utilities, text cleaning, metrics, routing harness, device selection, results registry, serving adapter) goes in **`src/utils/`**, and model folders import from it (`from src.utils.metrics import evaluate`). The empty folders `src/improved_model_2/` … `src/improved_model_5/` already exist as placeholders. Leave them empty until their section is started, and add an `__init__.py` only then.
3. **Artifacts follow the baseline_model_2 convention:**
   - **Metrics and the small served model** go to `docs/metrics/improved_model_N/`.
   - **Large caches** (Parquet splits, embeddings `.npy`, word vectors `.kv`, transformer checkpoints) go to `data/cache/improved_model_N/`, which is gitignored.
4. **Model inputs:** `complaint_text` only. Never feed `product`, `sub_product`, `issue`, `sub_issue`, `team_*`, `product_issue_*`, IDs, group hashes, `company` or `source_*` columns to a model.
5. **Fit on train only.** Vectorizers, embedding models, class weights and scalers are fit on train. Calibration and thresholds are fit on validation. The test set and the 2026 cohort are only scored.
6. **Leakage:** a `leakage_group_id` must never span two splits (same rule as baseline 2: groups crossing a period boundary are dropped entirely).
7. **Reproducibility:** `RANDOM_SEED = 42` everywhere. Log library versions (including `sklearn.__version__`) to `run_info.json`.
8. **Git hygiene:**
   - Work on a branch per model (`improved_model_1`, …) created from the current HEAD, which already contains the latest `main`.
   - The working tree currently shows ~47 "modified" files that are **only line-ending (CRLF) differences**. Check with `git diff --ignore-all-space --stat`. The two LFS zips also show as modified because Git LFS isn't set up locally; run `git lfs install` once.
   - **Never `git add -A` or `git add .`.** Stage only the files this plan creates or edits, by path.
   - Never commit data, `data/cache/`, embeddings or checkpoints.
   - `*.joblib` is **not** blanket-ignored, because baseline 2's `model.joblib` is committed for the app. Commit an improved model's `model.joblib` only if it is the one served by the app (§5) and ≤ 95 MB.
   - Add to `.gitignore`: `data/cache/`, `*.npy`, `*.kv`, `**/checkpoints/`.
9. **Memory:** the full CSV is 1.9 GB. Read it with `usecols=[...]`, `dtype={"complaint_id": "string"}`, `keep_default_na=False` and `chunksize=100_000`. Convert it to Parquet once (§1.3) and read the Parquet file afterwards.
10. **Housekeeping:** delete `data/processed/splits/w2v_corpus.csv`, a partial scratch file from an exploratory run. Touch nothing else in that folder. Its other CSVs are stale pre-session-05 splits without `split_info.json`, and baseline 2 will rebuild them itself if its owner reruns it.
11. **Dependencies:** `requirements.txt` is what **Streamlit Community Cloud installs**, so keep it limited to what the app needs at runtime. Put training-only dependencies (`torch`, `gensim`, `sentence-transformers`, `transformers`, `datasets`, `accelerate`, `peft`, `captum`) in a new **`requirements-train.txt`**. **Do not upgrade scikit-learn** (the `.venv` has 1.9.1, and baseline 2's `model.joblib` was saved with it).
12. **GPU available:** use it for every neural step (sentence embeddings, BERT fine-tuning, DAPT). Always select the device via `src/utils/device.py` (§0.1), never hard-code `"cuda"`, so the code still runs on a CPU-only teammate's laptop.

### 0.1 GPU setup (one-time, Windows)
1. **The current `.venv` has `torch 2.14.0+cpu`**, a CPU-only build. It must be replaced by a CUDA build.
2. Check the driver and CUDA version: `nvidia-smi`.
3. Reinstall torch as the CUDA build matching the driver (pick the command for your CUDA version and Python 3.14 at pytorch.org). For example:
   ```powershell
   python -m pip install --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cu128
   ```
   `--no-deps` keeps pip from touching other packages (scikit-learn stays at 1.9.1). Document the command in the README. The torch line in `requirements-train.txt` stays unpinned to a CUDA build.
4. Verify:
   ```powershell
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   The version must end in `+cu…`, not `+cpu`.
5. `src/utils/device.py`:
   ```python
   def get_device(prefer: str = "auto") -> str        # "cuda" if available (and prefer != "cpu") else "cpu"
   def gpu_info() -> dict                             # name, total VRAM GB, torch/CUDA versions → saved in run_info.json
   def amp_dtype() -> torch.dtype | None              # bf16 if torch.cuda.is_bf16_supported() else fp16 on GPU; None on CPU
   def suggested_batch_size(vram_gb, task) -> int     # simple table below
   ```
   Import torch lazily inside these functions so `src.utils` stays importable in the Streamlit app without torch.

   | Task (max_len) | ≤8 GB VRAM | 12–16 GB | ≥24 GB |
   |---|---|---|---|
   | Sentence encoding, base model (512) | 64 | 128 | 256 |
   | Fine-tune BERT/DeBERTa-base (512) | 8 + grad-accum 4 | 16 + grad-accum 2 | 32 |
   | Fine-tune ModernBERT-base (1024) | 4 + grad-accum 8 | 8 + grad-accum 4 | 16 + grad-accum 2 |

   Enable gradient checkpointing if batch size 8 still runs out of memory.
6. Windows notes: use `num_workers=0` in DataLoaders (or wrap entry points in `if __name__ == "__main__":`), and set `TOKENIZERS_PARALLELISM=false` to avoid warnings.
7. The scikit-learn models (NB, LinearSVC, LogisticRegression) run on **CPU**. That is fine, since they train in minutes. Do not add cuML (it is not supported on native Windows).

### 0.2 What already exists (read these before coding)

| Path | What it is | How this plan uses it |
|---|---|---|
| `streamlit_app.py` | Consumer intake app (Submit / Track tabs). Calls `src.baseline_model_2.predict.predict(text, model=...)`, stores the result, and shows the team plus a tracking ID. | §5 swaps in a serving adapter; the default model stays baseline 2 |
| `src/app/storage.py`, `src/app/schema.sql` | Supabase store (deployed) with local SQLite fallback (`data/app/complaints.db`). Columns: `tracking_id, submitted_at, complaint_text, team_id, team_name, confidence, status`. | §5 adds nullable issue/route/model columns via a migration |
| `src/baseline_model_2/` | Production team model and its split builder (temporal, crossing groups dropped, train capped at 3,000 per team) plus the north-star evaluation harness (`coverage_curve`, `threshold_for`, `bootstrap_ci`, `per_team_routing`, ECE with 10 bins) | Reference model. Its protocol and harness are replicated in `src/utils/` |
| `docs/metrics/baseline_model_2/` | Committed artifacts: `model.joblib` (loaded by the app), `north_star.json`, `per_team_routing.csv`, `coverage_curve.csv`, `predictions_test.csv` (all 109,834 test IDs) | Parity check (§1.6) and the comparison baseline |
| `data/preparation/advanced_text.py` | `clean_text()`, the team's canonical preprocessing: NFKC, HTML unescape, lowercase, negation-preserving contractions, URL/email removal, `xx…` redaction removal, ≥6-digit numbers → `number` | Default preprocessor for all improved models |
| `data/preparation/README.md` | Official full temporal partitions: train 774,412 / val 153,502 / test 109,834; 6,867 records in 1,949 crossing groups excluded; smallest issue support 374 / 54 / 30 | `build_splits` must reproduce these counts |
| `data/complaint_router_2026_holdout.zip` | Jan–Feb 2026 cohort (29,798 complaints; 72 issues, no `T01_I003` examples); `fresh_holdout_2026_01_02.parquet`. **Already used for evaluation, not an untouched test.** | Secondary "forward drift" check only; never trained on |
| `src/baseline_model_1/`, `src/data/` | Earlier JPMorgan-only product classifier and its pipeline | Untouched and not used |
| `reports/session05.md` | Next goals: issue prediction alongside team, UI improvements, end-to-end product, PII obfuscation before storing or showing | §5 and the roadmap cover these |

---

## 1. Shared foundation: `src/utils/`

```
src/utils/
├── __init__.py
├── config.py          # paths, seed, split dates, caps, column names, routing constants (copied values from baseline_model_2)
├── labels.py          # team/issue mappings, issue canonicalization, team→issues mask
├── text.py            # re-exports data.preparation.advanced_text.clean_text; tokenizer for word vectors
├── build_splits.py    # CSV → Parquet, temporal + group-aware splits identical to baseline 2's val/test
├── data.py            # load_split(name), load_holdout_2026()
├── device.py          # get_device(), gpu_info(), amp_dtype(), suggested_batch_size() (§0.1)
├── metrics.py         # classification metrics (team, issue, joint, oracle, hard pairs, bootstrap CIs)
├── routing.py         # north-star harness replicated from baseline_model_2 + per-team thresholds
├── registry.py        # append results to reports/model_comparison.csv
└── serving.py         # model adapter used by the Streamlit app (§5; create it in §5, not before)
tests/
└── test_parity_baseline2.py   # §1.6
```

Target repository layout after this plan:
```
src/
├── app/                  # existing app storage (extended in §5)
├── baseline_model_1/     # existing, untouched
├── baseline_model_2/     # existing production model, untouched
├── data/                 # existing baseline-1 pipeline, untouched
├── experiments/          # existing probe script
├── utils/                # shared, reusable (this section)
├── improved_model_1/     # TF-IDF + LinearSVC, hierarchical (§2)
├── improved_model_2/     # placeholder (empty) → embeddings (§3)
├── improved_model_3/     # placeholder (empty) → fine-tuned BERT-base (§4)
├── improved_model_4/     # placeholder (empty) → DeBERTa/ModernBERT hierarchical multi-task (§4)
└── improved_model_5/     # placeholder (empty) → DAPT + calibration/conformal + ensemble (§4)
docs/metrics/improved_model_N/   # metrics + served model (like baseline_model_2)
data/cache/improved_model_N/     # large, gitignored
```

### 1.1 `config.py`
```python
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FULL_CSV = PROCESSED / "complaints_product_issues_2024_2025.csv"
CACHE = ROOT / "data" / "cache"
FULL_PARQUET = CACHE / "complaints_product_issues_2024_2025.parquet"
SPLITS_V2 = CACHE / "splits_v2"
HOLDOUT_ZIP = ROOT / "data" / "complaint_router_2026_holdout.zip"
HOLDOUT_PARQUET = CACHE / "holdout_2026" / "fresh_holdout_2026_01_02.parquet"
ISSUE_MAPPING = PROCESSED / "issue_mapping.csv"
TEAM_MAPPING = PROCESSED / "team_mapping.csv"
BASELINE2_METRICS = ROOT / "docs" / "metrics" / "baseline_model_2"
METRICS_ROOT = ROOT / "docs" / "metrics"          # improved models write to METRICS_ROOT / "improved_model_N"
REPORTS = ROOT / "reports"

TEXT_COL, TEAM_COL, ISSUE_COL = "complaint_text", "team_id", "issue_id"   # issue_id = canonical product_issue_id
GROUP_COL, DATE_COL, ID_COL = "leakage_group_id", "date_received", "complaint_id"
USECOLS = [ID_COL, DATE_COL, "company", TEXT_COL, TEAM_COL, "team_name", "product_issue_id", GROUP_COL]

RANDOM_SEED = 42
# Same boundaries as src/baseline_model_2/config.py (exclusive upper bounds)
TRAIN_END = "2025-07-01"; VAL_END = "2025-10-01"; TEST_END = "2026-01-01"
TRAIN_CAP_PER_ISSUE = 10_000      # main training set; None = all 774,412
BASELINE2_CAP_PER_TEAM = 3_000    # reproduces baseline 2's training recipe for a same-data comparison
MERGE_ISSUE_VARIANTS = True
ISSUE_VARIANT_MERGES = {"T01_I007": "T01_I008",   # "...existing issue" → "...existing problem"
                        "T01_I003": "T01_I001"}   # identity-theft monitoring wording variants (T01_I003 is absent from the 2026 cohort)

# Routing / north star — identical to baseline_model_2
PRECISION_TARGETS = (0.95, 0.90, 0.85)
TEAM_PRECISION_FLOOR = 0.85
MIN_ROUTED = 50
N_BOOTSTRAP = 1000
ECE_BINS = 10
```

### 1.2 `labels.py`
- `load_issue_mapping()` returns the DataFrame from `issue_mapping.csv`.
- `canonical_issue(pid)` applies `ISSUE_VARIANT_MERGES` when `MERGE_ISSUE_VARIANTS` is on.
- Write `data/processed/issue_canonical.csv` (`product_issue_id, issue_id, team_id, issue_label`). `data/processed/` is gitignored, so write a copy to `docs/metrics/improved_model_1/issue_canonical.csv` as well.
- `issue_to_team: dict[str, str]`, `team_to_issues: dict[str, list[str]]`, `team_names: dict[str, str]`.
- `team_issue_mask(issue_classes, team_classes) -> np.ndarray[bool]` with shape (n_teams, n_issues).

### 1.3 `build_splits.py` (run once: `python -m src.utils.build_splits`)
Reproduce the **official protocol** from `data/preparation/README.md` and `src/baseline_model_2/data_loader.py`, so val and test are exactly baseline 2's evaluation sets:
1. If `FULL_PARQUET` is missing, stream the CSV in chunks with `USECOLS` and write Parquet (pyarrow).
2. Add `issue_id = canonical_issue(product_issue_id)`.
3. **Temporal assignment by `date_received`:** train < 2025-07-01 ≤ val < 2025-10-01 ≤ test < 2026-01-01.
4. **Group rule (same as baseline 2):** any `leakage_group_id` present in more than one period is **dropped entirely** from all partitions. Expect 6,867 records in 1,949 groups.
5. The uncapped partitions must match the README: **train 774,412, val 153,502, test 109,834**. Assert these counts.
6. Training sets derived from the uncapped train partition:
   - `train_cap_issue`: at most `TRAIN_CAP_PER_ISSUE` rows per `issue_id` (`groupby("issue_id").sample(..., random_state=RANDOM_SEED)`, or all rows when below the cap). About 259k rows. This is the **main** training set.
   - `train_3k_team`: `sample(frac=1, random_state=42).groupby("team_id").head(3000)`, the same recipe as baseline 2 (32,599 rows). Used for the same-data comparison.
   - `train_full_ids.parquet`: IDs of all 774,412 rows. Model 2 uses them for unsupervised word-vector training.
7. Val and test keep their **natural distribution** (no capping).
8. **Parity check against baseline 2:** the test `complaint_id` set must equal the IDs in `docs/metrics/baseline_model_2/predictions_test.csv` exactly (109,834). Assert it. Val size must be 153,502, and per-team val counts should match the baseline 2 doc.
9. Write `SPLITS_V2/{train_cap_issue,train_3k_team,val,test}.parquet` with columns `complaint_id, date_received, company, complaint_text, team_id, team_name, issue_id, leakage_group_id`.
10. Write `SPLITS_V2/split_report.json` containing:
    - rows per split, per team and per issue;
    - crossing-group rows dropped;
    - date ranges;
    - assertions: group sets disjoint, all 11 teams and all canonical issues in every split, and the parity result.

    Copy `split_report.json` to `docs/metrics/improved_model_1/` so it is committed.

### 1.4 `data.py`
- `load_split(name) -> pd.DataFrame` reads from `SPLITS_V2`.
- `load_holdout_2026() -> pd.DataFrame` extracts `fresh_holdout_2026_01_02.parquet` from `HOLDOUT_ZIP` into `data/cache/holdout_2026/` on first use (never into `data/processed/`). It adds `issue_id` via `canonical_issue` and verifies that the columns `complaint_id, complaint_text, team_id, product_issue_id` exist, printing the actual column list if they don't.

### 1.5 `text.py`
```python
from data.preparation.advanced_text import clean_text   # canonical, stable import path (pickled vectorizers reference it)
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
def tokenize(text: str) -> list[str]: return TOKEN_RE.findall(clean_text(text))
```
- Use `clean_text` **directly** as the `preprocessor=` of every `TfidfVectorizer`: a module-level function, not a lambda, so the model pickles and loads in the app.
- Do not copy or modify `data/preparation/advanced_text.py`.
- Note: `clean_text` **removes** `XXXX` redactions. The transformer models (§4) will instead map them to special tokens; that happens in their own tokenization step.

### 1.6 `metrics.py` and `routing.py`
`metrics.py`: `evaluate(y_team, team_proba, team_classes, y_issue=None, issue_scores=None, issue_classes=None, oracle_issue_scores=None) -> dict`:

| Group | Metrics |
|---|---|
| Team | accuracy, macro-F1, weighted-F1, per-team P/R/F1/support, confusion matrix (11×11) |
| Issue | accuracy, macro-F1, weighted-F1, **top-3 accuracy**, per-issue P/R/F1/support |
| Joint | exact match (team **and** issue correct) |
| Oracle | issue accuracy / macro-F1 when the **true** team is given (ceiling for the hierarchy) |
| Uncertainty | bootstrap 95% CI (1,000 resamples) for team macro-F1 and issue macro-F1 |
| Hard pairs | recall of T02 (Debt Collection) and share of it predicted as T01; recall of T05 (Money Transfer) and share predicted as T04; recall of T11 and share predicted as T01/T02 |

`routing.py`: **replicate** baseline 2's north-star protocol exactly (copy the logic of `expected_calibration_error` (10 bins), `coverage_curve` (501 thresholds, `MIN_ROUTED=50`), `threshold_for`, `routed_stats`, `bootstrap_ci`, `per_team_routing` and `north_star_summary` from `src/baseline_model_2/train.py`, with a comment crediting the source). Then add:
- `per_team_thresholds(val_conf, val_pred, val_true, target=0.95)`: for each predicted team, the lowest threshold where precision of complaints routed **to that team** on validation is ≥ target (with `MIN_ROUTED`). A team that can't reach the target is never auto-routed. The baseline 2 doc lists this as its next step.
- `apply_per_team_thresholds(...)` → coverage, precision, per-team table, bootstrap CI on test.

Outputs use the **same file names and formats as baseline 2** so the two can be compared directly: `metrics.json`, `classification_report.json`, `confusion_matrix.csv`, `north_star.json`, `per_team_routing.csv`, `coverage_curve.csv`, `predictions_test.csv`. Add issue-level files next to them: `classification_report_issue.json`, `issue_metrics.json`.

**Parity test** (`tests/test_parity_baseline2.py`, run with `python -m pytest tests -q`):
1. Load `docs/metrics/baseline_model_2/predictions_test.csv` and the 0.872 threshold from `north_star.json`.
2. Recompute coverage, precision and `per_team_routing` with `utils.routing`.
3. Assert they equal the committed values: coverage 0.4646, precision 0.9444, and `per_team_routing.csv` row for row.
4. Also load baseline 2's `model.joblib` and score the rebuilt test split. Assert macro-F1 0.6475 ± 0.001, which proves the split and harness match.

### 1.7 `registry.py`
`log_result(model_name, split_name, metrics_dict)` appends one row to `reports/model_comparison.csv` with these columns:

`timestamp, model, train_set, eval_set, team_acc, team_macro_f1, issue_macro_f1, issue_top3, joint_exact, north_star_coverage_95, precision_at_95_threshold, per_team_thr_coverage_95, teams_below_floor, ece, n_eval, git_commit`

Seed it with a **baseline_model_2 row built from its committed JSON files**, without retraining. Every model folder calls `log_result`, so the comparison table grows with each improvement.

---

## 2. `src/improved_model_1/`: TF-IDF + Linear SVM, hierarchical team → issue

### 2.1 Why this design
A probe on the old balanced split (`src/experiments/team_vs_issue_probe.py`) gave:

| Strategy | Team macro-F1 | Issue macro-F1 |
|---|---:|---:|
| Direct team classifier | **0.780** | – |
| Flat 73-way issue, team derived via mapping | 0.741 | 0.338 |
| **Hierarchical: team, then per-team issue model** | **0.780** | **0.361** |
| Ceiling: perfect team → issue | 1.000 | 0.470 |

So the model **predicts the team first**, then predicts the **issue within the predicted team**, and reports issue as a **top-3 suggestion**. The team output keeps baseline 2's calibrated-confidence + threshold routing.

### 2.2 Folder layout
```
src/improved_model_1/
├── __init__.py
├── config.py        # model-specific hyperparameters, grids, artifact dir = docs/metrics/improved_model_1
├── features.py      # build_vectorizer()
├── model.py         # HierarchicalSVM class
├── train.py         # entry point: tune → fit → calibrate → thresholds → evaluate → save
└── predict.py       # CLI + importable predict(), output compatible with baseline 2
```
Artifacts go to `docs/metrics/improved_model_1/`.

### 2.3 `features.py`
```python
from src.utils.text import clean_text

def build_vectorizer(use_char=True, word_max=150_000, char_max=150_000):
    word = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=3, max_df=0.95,
                           max_features=word_max, sublinear_tf=True, strip_accents="unicode", dtype=np.float32)
    if not use_char:
        return word
    char = TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                           max_features=char_max, sublinear_tf=True, dtype=np.float32)
    return FeatureUnion([("word", word), ("char", char)])
```
If memory is tight on ~259k rows, set `use_char=False` or `TRAIN_CAP_PER_ISSUE=5000` and note it in the doc.

### 2.4 `model.py`: `HierarchicalSVM`
- `team_clf = LinearSVC(C=C_team, class_weight="balanced", dual="auto", max_iter=5000)`
- `issue_clfs = {team: LinearSVC(C=C_issue, class_weight="balanced", ...)}`: one per team, trained only on that team's rows. A team with a single issue gets a constant predictor.
- `flat_issue_clf`: an **ablation only**, a single LinearSVC over all issues.
- `fit(X, y_team, y_issue)`
- `team_proba(X)` → calibrated (n, 11) probabilities. `predict_team(X)`.
- `predict_issue_topk(X, teams, k=3)`: for each row, use the issue classifier of the given team (predicted at inference, true team for the oracle metric) and return the top-k issue IDs with probabilities.
- **Calibration:** use **isotonic**, as baseline 2 does. Wrap the fitted team classifier with `CalibratedClassifierCV(FrozenEstimator(team_clf), method="isotonic")` and fit it on the **validation** set (153,502 rows at the real mix; this also corrects the class prior learned from the capped training set). For each per-team issue classifier, fit an isotonic calibrator on that team's validation rows when it has ≥ 2 issues and ≥ 200 rows. Otherwise use sigmoid, or softmax over decision scores as a last resort.
- **Artifact size budget (the app loads this file from Git):** `model.joblib` ≤ 95 MB (GitHub's hard limit is 100 MB).
  - Save with `joblib.dump(..., compress=3)` and cast `coef_`/`intercept_` to float32 before saving.
  - If it is still too big, lower `word_max`/`char_max`, or give the issue models word-only features.
  - Record the final size in `run_info.json`.

### 2.5 `train.py`
Run with `python -m src.improved_model_1.train [--train-set cap_issue|3k_team] [--no-char] [--val-protocol shared|split] [--quiet]`.

1. Load `train_*`, `val` and `test` from `SPLITS_V2`, and the 2026 cohort via `load_holdout_2026()`.
2. Fit the vectorizer on train. Transform val, test and the 2026 cohort.
3. **Tune on val** (grid in `config.py`): `C_team ∈ {0.1, 0.25, 0.5, 1.0}` by val team macro-F1, then `C_issue ∈ {0.1, 0.25, 0.5, 1.0}` by val issue macro-F1 under the oracle team. Log the grid to `tuning.json`.
4. Refit on train with the best C values.
5. Calibrate on val (2.4). `--val-protocol`:
   - `shared` (default) matches baseline 2: calibration and thresholds both fit on the full validation set.
   - `split` fixes baseline 2's documented limitation: a group-aware 50/50 split of validation, calibrating on half A and choosing thresholds on half B. Report both protocols.
6. **Routing:** compute the global north-star threshold on validation and apply it frozen to test. Compute per-team thresholds (§1.6) the same way. Report both, and the per-team guardrail (floor 0.85).
7. Evaluate on **test** (primary, same 109,834 complaints as baseline 2) and on the **2026 cohort** (secondary: "forward drift check on an already-used cohort", never used for any choice). Evaluate these models:
   - `improved_model_1`: hierarchical, calibrated, trained on `train_cap_issue`.
   - `improved_model_1_3k`: same model trained on `train_3k_team`. Same data as baseline 2, so it isolates the effect of the classifier from the effect of more data. This answers the open question in session 05.
   - `flat_issue_ablation`: flat LinearSVC, team looked up from the issue.
   - `nb_reference_cap_issue`: baseline 2's recipe (TF-IDF 1–2, MultinomialNB α=0.1, `fit_prior=False`, balanced sample weights, isotonic on val) retrained on `train_cap_issue`. Reimplement it in ~10 lines; do not import from `baseline_model_2`.
   - `baseline_model_2`: the committed numbers (registry seed row). Its row on the 2026 cohort is computed by loading its `model.joblib` read-only.
   - `dummy_most_frequent`.
8. Save to `docs/metrics/improved_model_1/`:
   - `model.joblib` (vectorizer + HierarchicalSVM + calibrators + thresholds)
   - all baseline-2-format files (§1.6), plus the issue files
   - `predictions_test.csv`, with baseline 2's columns plus `issue_top1, issue_top2, issue_top3, actual_issue_id, auto_routed_per_team`
   - `run_info.json` (versions, `gpu_info()` is not needed here, train set, C values, row counts, runtime, artifact size, thresholds)
9. Call `registry.log_result` for every model × eval set.

### 2.6 `predict.py`
Output must be a **superset of `src.baseline_model_2.predict.predict()`**, so the app's `build_record()` keeps working unchanged:
```python
def load_model(path=ARTIFACTS_DIR / "model.joblib"): ...
def predict(text: str, top_k: int = 3, model=None) -> dict
```
```json
{"predicted_team_id": "T02", "predicted_team_name": "Debt Collection Team", "confidence": 0.91,
 "top_k": [{"team_id": "T02", "team_name": "Debt Collection Team", "confidence": 0.91}, ...],
 "issue_top_k": [{"issue_id": "T02_I001", "issue_label": "Attempts to collect debt not owed", "confidence": 0.64}, ...],
 "route": "auto",
 "model_name": "improved_model_1"}
```
`route` is `"auto"` or `"review"`. It is `"auto"` when confidence ≥ the per-team threshold of the predicted team (falling back to the global threshold), both saved in the artifact.
CLI: `python -m src.improved_model_1.predict "Debt collector keeps calling about a loan I already paid" --top-k 3`.

### 2.7 Documentation
Write `docs/improved_model_1.md` in the same structure as `docs/baseline_model_2.md`:
- objective and design
- data (reusing the same split numbers)
- the model
- evaluation table **against baseline 2 on the identical test set**
- per-team table
- north star (global and per-team thresholds, with CIs)
- guardrail
- issue results (macro-F1, top-3, oracle ceiling)
- the 3k-vs-cap comparison
- hard pairs
- 2026 cohort
- known limitations
- how to run and generated artifacts

Also add a short "Improved Model 1" section to the top-level `README.md` under the Baseline 2 section. Do not change the Baseline 2 text.

### 2.8 Acceptance criteria
- [ ] `build_splits` passes all assertions, and the counts match `data/preparation/README.md`. The test IDs equal baseline 2's.
- [ ] `tests/test_parity_baseline2.py` passes.
- [ ] Team macro-F1 on test **>** 0.647 (baseline 2).
- [ ] North-star coverage at the validation-fit 95% threshold **≥** 46.5%, with test precision and CI reported honestly. Per-team thresholds are reported separately.
- [ ] Guardrail: every team that auto-routes stays ≥ 0.85 routed precision on test, or the breach is documented.
- [ ] Hierarchical issue macro-F1 **≥** flat ablation.
- [ ] `model.joblib` ≤ 95 MB, and `predict()` output contains every key baseline 2 returns.
- [ ] `reports/model_comparison.csv` has rows for baseline_model_2, dummy, NB reference, improved_model_1, improved_model_1_3k and the flat ablation, on test and on the 2026 cohort.

---

## 3. `src/improved_model_2/`: better text representations (embeddings)

Same hierarchy, labels, splits, calibration, routing harness, metrics and registry as model 1. **Only the features change.** Each variant below is one config, logged separately. The best variant on **val** becomes the official `improved_model_2`. Metrics go to `docs/metrics/improved_model_2/`; embeddings and word vectors go to `data/cache/improved_model_2/`.

### 3.1 Folder layout
```
src/improved_model_2/
├── config.py
├── embedders.py            # Word2VecEmbedder, FastTextEmbedder, SentenceEmbedder, cache helpers
├── train_word_vectors.py   # trains Word2Vec / FastText on train-period text only
├── encode_sentences.py     # caches sentence-transformer embeddings to .npy
├── train.py                # --variant {w2v_mean, w2v_tfidf, ft_tfidf, sbert, hybrid_tfidf_w2v, hybrid_tfidf_sbert}
└── predict.py              # same output contract as model 1
```
Dependencies go in **`requirements-train.txt`**, not `requirements.txt`: `gensim` (not yet installed), `sentence-transformers` (6.1 already in `.venv`), `torch` (CUDA build, §0.1).

### 3.2 Variants

| Variant | Representation | Classifier |
|---|---|---|
| `w2v_mean` | Word2Vec (CBOW, 300d, window 8, min_count 5, 5 epochs) trained on **all 774,412 train-period** texts (`train_full_ids`, never val/test); mean of word vectors | StandardScaler → LogisticRegression (balanced, max_iter 3000) |
| `w2v_tfidf` | Same vectors, **IDF-weighted** mean (IDF from train) | same |
| `ft_tfidf` | gensim **FastText** (subword n-grams 3–6, 300d), IDF-weighted mean; handles typos like "dipute" and "zell" | same |
| `sbert` | Frozen sentence embeddings: `BAAI/bge-base-en-v1.5` (default) and `intfloat/e5-base-v2` (prefix `"query: "`), `max_seq_length=512`, `normalize_embeddings=True` | LogisticRegression |
| `hybrid_tfidf_w2v` | `hstack([TF-IDF from model 1, α · w2v_tfidf])`; tune α ∈ {0.05, 0.1, 0.2, 0.4} on val | LinearSVC (as model 1) |
| `hybrid_tfidf_sbert` | `hstack([TF-IDF, α · sbert])`, with α tuned the same way | LinearSVC |

Tokenization for word vectors: `src.utils.text.tokenize` (i.e. `clean_text`, then regex `[a-z]+(?:'[a-z]+)?`). Sentence encoders get the **raw** `complaint_text` with only whitespace normalized; they have their own tokenizers.

### 3.3 Practical notes
- **Run sentence encoding on the GPU.** `encode_sentences.py` takes these flags:
  - `--device {auto,cuda,cpu}` (default `auto` via `utils.device.get_device()`)
  - `--model`
  - `--batch-size` (default from `suggested_batch_size`)
  - `--max-len 512`
  - `--limit`

  Load the model in half precision on GPU (`model.half()` or `model_kwargs={"torch_dtype": amp_dtype()}`) and encode inside `torch.inference_mode()`. Cache to `data/cache/improved_model_2/emb_{model}_{split}.npy` (float16) so each split is encoded once.
- On GPU, encode **all** splits: train_cap_issue ≈259k, val 153,502, test 109,834, 2026 cohort 29,798. Expect minutes to tens of minutes per model. CPU fallback only if CUDA is unavailable: use `BAAI/bge-small-en-v1.5` at 256 tokens and state that in the doc.
- Sort texts by length before batching (restore the original order afterwards). With GPU this is the biggest speed-up.
- Word2Vec and FastText (gensim) are CPU-only; use `workers = os.cpu_count() - 1`. LogisticRegression and LinearSVC on the embeddings stay on CPU.
- Log `gpu_info()` and the encoding throughput (docs/s) into `run_info.json`.
- Expectation to write in the doc: averaged word vectors often **do not beat** TF-IDF on long documents. The expected gains come from the **hybrid** variants and frozen sentence embeddings. Report whatever happens.
- Save the Word2Vec nearest neighbours for 10 domain words (`repossession, zelle, fdcpa, collector, dispute, chargeback, escrow, overdraft, garnishment, forbearance`) to `docs/metrics/improved_model_2/w2v_neighbors.json` for the report.
- **Serving note:** the `sbert` and hybrid-sbert variants need torch and a 400+ MB encoder at inference. They are **not** servable from Streamlit Community Cloud as-is (§5.5). Word-vector variants are, if the `.kv` file is small enough or the vectors are pruned to the training vocabulary.

### 3.4 Acceptance criteria
- [ ] All 6 variants logged in `reports/model_comparison.csv` (val, test and the 2026 cohort).
- [ ] `docs/improved_model_2.md` with a table comparing baseline 2 → model 1 → each model-2 variant, and the chosen variant justified **by validation**, not test.

---

## 4. Roadmap for later folders (spec to be expanded when started)

The folders `src/improved_model_3/`, `4/` and `5/` exist but stay **empty** until their specs are written. Their metrics go to `docs/metrics/improved_model_N/`; checkpoints go to `data/cache/improved_model_N/checkpoints/`.

| Folder | Improvement | GPU usage |
|---|---|---|
| `src/improved_model_3/` | **Fine-tuned BERT-base** (`bert-base-uncased`), team head first, then the issue head. Hugging Face `Trainer`; freeze → gradual unfreeze (heads → top ⅓ layers → all with layer-wise LR decay 0.85); max_len 512 with head+tail truncation; class-weighted CE; special tokens `[REDACTED]`, `[AMOUNT]`, `[DATE]` (map `XXXX` runs, `{$…}` and `XX/XX/XXXX` to them before tokenizing). Calibrate with temperature scaling on val; same routing harness. | Required. AMP (`bf16=True` if supported, else `fp16=True`), batch size from §0.1, gradient checkpointing if OOM, `eval_strategy="steps"` on a 20k val subset for speed |
| `src/improved_model_4/` | **DeBERTa-v3-base and ModernBERT-base** with **hierarchical multi-task heads** (shared encoder; team head + issue head masked to the team's issues; loss = CE_team + 0.5·CE_issue). ModernBERT at max_len 1024. LoRA (r=16) vs full fine-tuning ablation. | Required. DeBERTa-v3 is unstable in fp16, so use bf16 or fp32 for it |
| `src/improved_model_5/` | **Domain-adaptive pretraining** (continued MLM on the 774,412 train-period complaint texts, 1 epoch, 30% masking) → re-run the model 4 recipe; then **calibration + selective routing** (temperature scaling, per-team thresholds, conformal prediction sets) and a logit **ensemble** with the model-1 TF-IDF SVM. | Required. DAPT is the longest GPU job, so checkpoint every N steps and resume with `resume_from_checkpoint` |
| later | **PII scrubbing** (Presidio + regex) in `src/utils/pii.py`, applied in the app **before storing or showing** a complaint (session-05 goal); LLM adjudicator for the low-confidence "review" queue; ONNX / distilled model export for cheap serving. | – |

Common rules for the GPU folders:
- Set seeds via `transformers.set_seed(42)`.
- Log `gpu_info()`, peak VRAM (`torch.cuda.max_memory_allocated()`), wall-clock training time and inference throughput to `run_info.json`.
- Save only the best checkpoint (`save_total_limit=1`, `load_best_model_at_end=True`, metric = val team macro-F1).

Every folder follows the same contract:
- `train.py`
- `predict.py` (output contract §2.6)
- metrics in `docs/metrics/improved_model_N/`
- `docs/improved_model_N.md`
- rows appended to `reports/model_comparison.csv`

---

## 5. App integration (after a model passes its acceptance criteria)

The Streamlit app (`streamlit_app.py`) keeps serving **baseline_model_2 by default** until an improved model beats it on the §2.8 criteria **and** the user approves the switch. Frontend changes are additive and backward-compatible.

### 5.1 `src/utils/serving.py`
```python
MODELS = {
    "baseline_model_2": ("src.baseline_model_2.predict", "load_model", "predict"),
    "improved_model_1": ("src.improved_model_1.predict", "load_model", "predict"),
}
def active_model_name() -> str                 # env CLARICASE_MODEL, else st.secrets["model"]["name"], else "baseline_model_2"
def load_router(name: str | None = None)       # returns (name, model) via importlib; cached by the caller
def route(text: str, name: str, model, top_k: int = 3) -> dict
```
`route()` always returns baseline 2's keys, plus `issue_top_k`, `route` and `model_name` when the model provides them (with `None` defaults otherwise). No torch import at module level.

### 5.2 `streamlit_app.py` (minimal edit)
- Replace `from src.baseline_model_2.predict import load_model, predict` with the `serving` adapter. `get_model()` caches `load_router()`, and `handle_submit` calls `route(...)`.
- On success, keep the current team message. If an issue is available, add one line: *"We understood this as: {issue_label}"*.
- If `route == "review"`, set status to `"Under review"` instead of `"Received"`.
- The Track tab shows the issue label when present.
- No other UI changes in this phase. The larger UI work in session 05's goals is a separate task.

### 5.3 Storage (`src/app/storage.py`, new `src/app/migrations/002_add_issue_columns.sql`)
- New **nullable** columns: `issue_id text, issue_label text, route text, model_name text`.
- Supabase: write the migration as `alter table public.complaints add column if not exists ...`. The user runs it once in the SQL editor. Do not edit the original `schema.sql`; reference the migration from it in a comment.
- `LocalStore`: on init, check `PRAGMA table_info(complaints)` and `ALTER TABLE ... ADD COLUMN` for any missing columns, so existing `data/app/complaints.db` files keep working.
- `build_record(text, prediction)`: fill the new fields with `prediction.get(...)` so baseline 2 predictions still work. `COLUMNS` includes the new fields.
- Until the Supabase migration has been run, `SupabaseStore.add` must not send the new keys. Gate them behind a `store.supports_issue_fields` flag, checked once by selecting the columns and catching the error.

### 5.4 Tests
Add `tests/test_app_contract.py`:
- `route()` output from both baseline 2 and improved model 1 contains every key `build_record()` needs.
- `LocalStore` migrates an old-schema SQLite file created in a temp dir.

### 5.5 Deployment constraints (Streamlit Community Cloud)
- The app loads the model from Git. Commit `docs/metrics/improved_model_1/model.joblib` (≤ 95 MB) only when switching to it.
- Streamlit Cloud installs `requirements.txt`, and model 1 needs only scikit-learn, so no change is required. Check that `supabase` is listed there, since the deployed app imports it: the current `requirements.txt` does not list it. Flag this to the user rather than silently changing deploy behaviour.
- Community Cloud has **no GPU and about 1 GB of RAM**, so transformer models (3–5) cannot be served there as-is. Options for later: export to ONNX + int8 quantization, distil into a small encoder, or host inference elsewhere (e.g. a Hugging Face Space or a small GPU endpoint) behind the same `route()` contract.
- To switch models: set `CLARICASE_MODEL = "improved_model_1"` (or add `[model] name = "improved_model_1"` in Streamlit secrets). Reverting is the same setting back to `baseline_model_2`.

---

## 6. Execution order for Claude Code

1. Run `git lfs install`. Confirm with `git diff --ignore-all-space --stat` that the pending "modifications" are line endings only, and do not stage them. Create branch `improved_model_1` from the current HEAD.
2. Delete `data/processed/splits/w2v_corpus.csv`. Update `.gitignore` (rule 0.8). Create `requirements-train.txt` (rule 0.11). Reinstall CUDA torch and verify (§0.1); record the output in `docs/improved_model_1.md` (environment section). Leave `src/improved_model_2/`–`5/` empty.
3. Implement `src/utils/` (§1, everything except `serving.py`) and run `python -m src.utils.build_splits`. Check the counts against `data/preparation/README.md` and the parity with baseline 2's test IDs.
4. Write and pass `tests/test_parity_baseline2.py`. **Stop and report if parity fails.**
5. Implement `src/improved_model_1/` (§2). Run `python -m src.improved_model_1.train` (cap_issue, shared protocol), then with `--train-set 3k_team`, then with `--val-protocol split`.
6. Write `docs/improved_model_1.md` and the README section. Commit code, metrics JSON/CSV, docs and tests by explicit paths. Do not commit `model.joblib` yet.
7. **Stop and show the user the comparison table vs baseline 2.** Ask whether to (a) integrate model 1 into the app (§5) or (b) continue to model 2.
8. If (a): implement §5 on branch `app/improved-model-1`, run the tests, and hand the Supabase migration SQL to the user. Commit `model.joblib` only after the user approves the switch.
9. Model 2: create branch `improved_model_2` and implement §3: run `train_word_vectors`, then `encode_sentences --device auto` (GPU), then `train --variant ...` for each variant. Write `docs/improved_model_2.md`. Commit. Stop and report before starting model 3.
