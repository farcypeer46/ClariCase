# 03 — Improved Model 2: Better Text Representations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Same hierarchy, calibration, routing and metrics as model 1 — swap the
**feature representation**. Compare six variants (Word2Vec mean, Word2Vec-IDF,
FastText-IDF, sentence-BERT, TF-IDF+W2V hybrid, TF-IDF+sBERT hybrid) and pick a
winner by validation macro-F1.

**Architecture:** Reuse `HierarchicalSVM` from model 1 with a different upstream
transformer. Word vectors are trained on **train-period text only** (all 774,412
rows) via gensim. Sentence-transformer embeddings are precomputed to `.npy`
cache once per split. Hybrids `hstack` TF-IDF with a scaled embedding block; α
tuned on val.

**Tech Stack:** gensim (Word2Vec, FastText), sentence-transformers,
torch (CUDA), scikit-learn (LogisticRegression, LinearSVC, StandardScaler,
FeatureUnion), scipy.sparse, numpy.

**Prerequisites:**
- Plans 00, 01, 02 complete.
- Model 1 has passed its acceptance criteria and been merged (or its branch is
  parked).
- CUDA torch installed and verified (plan 00 Task 5).
- Fresh branch: `git checkout -b improved_model_2` from the current HEAD.
- `requirements-train.txt` extras installed:
  `pip install -r requirements-train.txt` (installs gensim; sentence-transformers
  may already be present).

---

## File layout produced

```
src/improved_model_2/
├── __init__.py
├── config.py                # variant grid, alpha grid, encoder names, cache paths
├── embedders.py             # Word2VecEmbedder, FastTextEmbedder, SentenceEmbedder
├── train_word_vectors.py    # gensim W2V + FastText on train_full_ids
├── encode_sentences.py      # CLI: cache SBERT embeddings per split
├── train.py                 # --variant {w2v_mean, w2v_tfidf, ft_tfidf, sbert, hybrid_tfidf_w2v, hybrid_tfidf_sbert}
└── predict.py               # output contract same as model 1
docs/metrics/improved_model_2/
├── w2v_neighbors.json
├── <one subfolder per variant>/{metrics.json, north_star.json, per_team_routing.csv, ...}
├── comparison.md            # variant → val macro-F1 table
└── model.joblib             # only the CHOSEN variant, ≤ 95 MB, torch-free (see §3.5)
data/cache/improved_model_2/      # gitignored
├── w2v.kv, w2v_full.model
├── fasttext.model
├── emb_bge-base_train.npy, emb_bge-base_val.npy, emb_bge-base_test.npy,
│   emb_bge-base_holdout.npy   (float16)
└── emb_e5-base_*.npy
docs/improved_model_2.md
```

---

### Task 1: Package skeleton + config

- [ ] **Step 1: Create files**

```python
# src/improved_model_2/__init__.py
"""Embedding-based feature variants (word2vec, fasttext, sBERT, hybrids)."""
```

```python
# src/improved_model_2/config.py
from src.utils.config import METRICS_ROOT, CACHE

ARTIFACT_DIR = METRICS_ROOT / "improved_model_2"
CACHE_DIR = CACHE / "improved_model_2"

VARIANTS = ("w2v_mean", "w2v_tfidf", "ft_tfidf", "sbert",
            "hybrid_tfidf_w2v", "hybrid_tfidf_sbert")

# Word vector training
W2V_DIM = 300
W2V_WINDOW = 8
W2V_MIN_COUNT = 5
W2V_EPOCHS = 5
W2V_SG = 0            # CBOW

FT_DIM = 300
FT_MIN_N = 3
FT_MAX_N = 6

# Sentence encoders (name → HF id, prefix)
SBERT_MODELS = {
    "bge-base": ("BAAI/bge-base-en-v1.5", None),
    "e5-base":  ("intfloat/e5-base-v2",  "query: "),
}
SBERT_DEFAULT = "bge-base"
SBERT_MAX_LEN = 512

# Hybrid alpha grid (val-tuned)
ALPHA_GRID = (0.05, 0.1, 0.2, 0.4)

# Domain probe words for Word2Vec neighbors report
DOMAIN_PROBE_WORDS = (
    "repossession", "zelle", "fdcpa", "collector", "dispute", "chargeback",
    "escrow", "overdraft", "garnishment", "forbearance",
)
```

- [ ] **Step 2: Commit**

```bash
cd F:/NLP_MSML641 && git add src/improved_model_2/__init__.py \
    src/improved_model_2/config.py
git commit -m "feat(improved_model_2): package skeleton and config"
```

---

### Task 2: `embedders.py`

- [ ] **Step 1: Test on synthetic corpus**

`tests/test_improved_model_2_embedders.py`:
```python
import numpy as np
import pytest
from src.improved_model_2.embedders import Word2VecEmbedder, FastTextEmbedder


@pytest.fixture
def toy_corpus():
    return [["debt", "collector", "call"], ["collector", "call", "again"],
            ["dispute", "charge", "bank"], ["bank", "account", "close"]]


def test_word2vec_embedder_returns_mean_vector(toy_corpus):
    emb = Word2VecEmbedder(dim=16, min_count=1, epochs=5).fit(toy_corpus)
    vec = emb.transform([["debt", "collector"], ["bank"]])
    assert vec.shape == (2, 16)
    assert np.isfinite(vec).all()


def test_word2vec_idf_weighted(toy_corpus):
    emb = Word2VecEmbedder(dim=16, min_count=1, epochs=5).fit(toy_corpus)
    emb.fit_idf([" ".join(t) for t in toy_corpus])
    v = emb.transform([["debt", "collector"]], weight="idf")
    assert v.shape == (1, 16)
```

- [ ] **Step 2: Implement `embedders.py`**

Three classes with a common `.fit(tokens_iter).transform(tokens_iter)` API:
- `Word2VecEmbedder(dim, window, min_count, epochs, sg)` — wraps
  `gensim.models.Word2Vec`. `.fit_idf(text_iter)` fits a `TfidfVectorizer` in
  `.vocabulary_` mode to obtain IDF weights.
- `FastTextEmbedder(...)` — wraps `gensim.models.FastText` (handles OOV via
  subwords); same interface.
- `SentenceEmbedder(model_name, prefix, device, batch_size, max_len)` —
  wraps `sentence_transformers.SentenceTransformer` with:
  - `.encode(texts, cache_path=None)` — sorts by length, encodes in half
    precision under `torch.inference_mode()`, restores order, caches to
    `.npy` (float16).
  - `.transform(texts)` returns the (N, dim) matrix.

- [ ] **Step 3: Verify unit tests**

Run: `pytest tests/test_improved_model_2_embedders.py -q`

- [ ] **Step 4: Commit**

```bash
cd F:/NLP_MSML641 && git add src/improved_model_2/embedders.py \
    tests/test_improved_model_2_embedders.py
git commit -m "feat(improved_model_2): Word2Vec / FastText / Sentence embedders"
```

---

### Task 3: `train_word_vectors.py`

- [ ] **Step 1: Implement**

```python
"""Train Word2Vec and FastText on the 774,412 train-period texts (never val/test).

Usage:
    python -m src.improved_model_2.train_word_vectors [--epochs 5]
"""
import argparse, json, os
from pathlib import Path

import pandas as pd
from src.improved_model_2.config import (
    CACHE_DIR, DOMAIN_PROBE_WORDS, FT_DIM, FT_MAX_N, FT_MIN_N,
    W2V_DIM, W2V_EPOCHS, W2V_MIN_COUNT, W2V_SG, W2V_WINDOW, ARTIFACT_DIR,
)
from src.improved_model_2.embedders import Word2VecEmbedder, FastTextEmbedder
from src.utils import config as C
from src.utils.data import load_split
from src.utils.text import tokenize


def _train_corpus_iter():
    # train_full_ids has just IDs; join back to the full train slice by
    # rebuilding from the temporal window in the Parquet.
    ids = pd.read_parquet(C.SPLITS_V2 / "train_full_ids.parquet")[C.ID_COL]
    full = load_split("train_cap_issue")  # smaller start; if incomplete, read all
    # If train_full_ids ⊄ train_cap_issue (it will be a superset), read Parquet directly:
    return _read_full_train_tokens(ids)


def _read_full_train_tokens(ids):
    # Streaming read from the Parquet cache; if size is a concern, load in chunks.
    p = C.SPLITS_V2 / "train_full.parquet"  # add in build_splits if missing
    df = pd.read_parquet(p, columns=[C.ID_COL, C.TEXT_COL])
    df = df[df[C.ID_COL].isin(set(ids))]
    for t in df[C.TEXT_COL]:
        yield tokenize(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=W2V_EPOCHS)
    args = ap.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tokens = list(_train_corpus_iter())
    w2v = Word2VecEmbedder(dim=W2V_DIM, window=W2V_WINDOW,
                           min_count=W2V_MIN_COUNT, epochs=args.epochs,
                           sg=W2V_SG, workers=max(os.cpu_count() - 1, 1))
    w2v.fit(tokens)
    w2v.save(CACHE_DIR / "w2v.kv", CACHE_DIR / "w2v_full.model")

    ft = FastTextEmbedder(dim=FT_DIM, window=W2V_WINDOW,
                          min_count=W2V_MIN_COUNT, min_n=FT_MIN_N, max_n=FT_MAX_N,
                          workers=max(os.cpu_count() - 1, 1))
    ft.fit(tokens)
    ft.save(CACHE_DIR / "fasttext.model")

    neighbors = {word: w2v.most_similar(word, top_k=10)
                 for word in DOMAIN_PROBE_WORDS if w2v.has(word)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "w2v_neighbors.json").write_text(
        json.dumps(neighbors, indent=2))


if __name__ == "__main__":
    main()
```

**Important:** update plan 01's `build_splits.py` to also write
`data/cache/splits_v2/train_full.parquet` (all 774,412 rows with text) if it
doesn't already. Add a one-line commit against plan 01's task 6 if so.

- [ ] **Step 2: Run**

```bash
F:/NLP_MSML641/.venv/Scripts/python -m src.improved_model_2.train_word_vectors
```
Expected: `w2v.kv`, `fasttext.model` in cache; `w2v_neighbors.json` in metrics.

- [ ] **Step 3: Commit**

```bash
cd F:/NLP_MSML641 && git add src/improved_model_2/train_word_vectors.py \
    docs/metrics/improved_model_2/w2v_neighbors.json
git commit -m "feat(improved_model_2): train W2V + FastText on train-period text"
```

---

### Task 4: `encode_sentences.py`

- [ ] **Step 1: Implement CLI**

```python
"""Cache sentence-transformer embeddings per split.

Usage:
    python -m src.improved_model_2.encode_sentences \
        --model bge-base --split train_cap_issue --device auto [--limit N]
"""
import argparse
import numpy as np

from src.improved_model_2.config import CACHE_DIR, SBERT_MAX_LEN, SBERT_MODELS
from src.improved_model_2.embedders import SentenceEmbedder
from src.utils import config as C
from src.utils.data import load_holdout_2026, load_split
from src.utils.device import get_device, gpu_info, suggested_batch_size


def _load(name):
    return load_holdout_2026() if name == "holdout_2026" else load_split(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SBERT_MODELS), required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=SBERT_MAX_LEN)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    info = gpu_info()
    vram = info.get("total_vram_gb", 8)
    bs = args.batch_size or suggested_batch_size(vram, "sbert")

    model_id, prefix = SBERT_MODELS[args.model]
    df = _load(args.split)
    if args.limit:
        df = df.head(args.limit)
    texts = df[C.TEXT_COL].astype(str).tolist()
    if prefix:
        texts = [prefix + t for t in texts]

    emb = SentenceEmbedder(model_name=model_id, device=get_device(args.device),
                           batch_size=bs, max_len=args.max_len)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"emb_{args.model}_{args.split}.npy"
    vecs = emb.encode(texts).astype(np.float16)
    np.save(out_path, vecs)
    print(f"[encode] {args.model} {args.split}: {vecs.shape} -> {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all 4 splits for the default model**

```bash
for split in train_cap_issue val test holdout_2026; do
  F:/NLP_MSML641/.venv/Scripts/python -m src.improved_model_2.encode_sentences \
      --model bge-base --split $split --device auto
done
```
Expected: 4 `.npy` files in `data/cache/improved_model_2/`. Each cached once.

- [ ] **Step 3: Commit**

```bash
cd F:/NLP_MSML641 && git add src/improved_model_2/encode_sentences.py
git commit -m "feat(improved_model_2): cache sentence-transformer embeddings per split"
```

---

### Task 5: `train.py` — variant-driven training loop

- [ ] **Step 1: Implement**

Reuses `HierarchicalSVM` from model 1. For each variant:

| Variant | Pre-classifier feature builder | Classifier |
|---|---|---|
| `w2v_mean` | mean of W2V word vectors | StandardScaler → LogisticRegression |
| `w2v_tfidf` | IDF-weighted mean of W2V vectors | same |
| `ft_tfidf` | IDF-weighted mean of FastText vectors | same |
| `sbert` | load `.npy` cache | LogisticRegression |
| `hybrid_tfidf_w2v` | `hstack([TF-IDF (from model 1), α · w2v_tfidf])`, α ∈ ALPHA_GRID | LinearSVC (`HierarchicalSVM`) |
| `hybrid_tfidf_sbert` | `hstack([TF-IDF, α · sbert])` | LinearSVC (`HierarchicalSVM`) |

For hybrids, tune α on val (val team macro-F1 via `HierarchicalSVM(C=1)`).
For W2V / SBERT variants, `HierarchicalSVM` runs against dense features
(`LinearSVC` handles dense fine; wrap in `csr_matrix` for consistency).

Reuse the calibrator/routing/write-artifacts code from model 1 — extract
`src/utils/train_helpers.py` if two places duplicate the same 200 lines.

For each variant, write artifacts to
`docs/metrics/improved_model_2/<variant>/` and call
`log_result(f"improved_model_2__{variant}", ...)` on val, test, holdout_2026.

- [ ] **Step 2: Run variants sequentially**

```bash
for v in w2v_mean w2v_tfidf ft_tfidf sbert hybrid_tfidf_w2v hybrid_tfidf_sbert; do
  F:/NLP_MSML641/.venv/Scripts/python -m src.improved_model_2.train --variant $v
done
```

- [ ] **Step 3: Pick the winner on val**

Read all variants' val macro-F1 (team), pick the highest, and:
- copy its artifacts up to `docs/metrics/improved_model_2/` (mirroring model
  1's layout), specifically `metrics.json`, `north_star.json`, etc.
- write `docs/metrics/improved_model_2/comparison.md` with the val table.

**Only if the winner is a word-vector variant** (torch-free at inference),
commit `docs/metrics/improved_model_2/model.joblib`. sBERT and hybrid-sBERT
variants are **not** servable from Streamlit Community Cloud (spec §5.5); note
this in the doc and skip `model.joblib`.

- [ ] **Step 4: Commit**

```bash
cd F:/NLP_MSML641 && git add src/improved_model_2/train.py \
    docs/metrics/improved_model_2/ reports/model_comparison.csv
git commit -m "feat(improved_model_2): six variants trained, winner selected on val"
```

---

### Task 6: `predict.py`

Same output contract as `src/improved_model_1/predict.py`, with
`"model_name": "improved_model_2__<variant>"`. If the winner is an sBERT
variant, `predict.py` lazily imports `sentence_transformers`.

- [ ] **Step 1: Implement**
- [ ] **Step 2: Skip-if-missing test** (mirrors model 1 task 6).
- [ ] **Step 3: Commit**

---

### Task 7: `docs/improved_model_2.md`

Compare **baseline 2 → model 1 → each model 2 variant** on val, test, and
holdout_2026. Include the W2V neighbours JSON as an appendix. Justify the
winner *by validation*, not test. State the serving constraint explicitly.

- [ ] **Step 1: Draft**
- [ ] **Step 2: Commit**

---

### Task 8: Push branch and stop

```bash
cd F:/NLP_MSML641 && git push -u origin improved_model_2
```

Report:
> "Model 2 done. Six variants logged in `reports/model_comparison.csv`.
> Winner on val: `<variant>` (macro-F1 `<x>`). Stop here — do not start model 3
> (§4) without a fresh spec expansion."

---

## Acceptance criteria (verbatim §3.4)

- [ ] All 6 variants logged in `reports/model_comparison.csv` for val, test,
  holdout_2026.
- [ ] `docs/improved_model_2.md` with a comparison table (baseline 2 → model 1
  → each variant) and a winner chosen **by validation**.

## Self-review

- [ ] Word vectors and sentence embeddings both **only trained on train-period
  text** (or the frozen pretrained encoder for sBERT). No leakage.
- [ ] `data/cache/improved_model_2/` is gitignored via plan 00 Task 3.
- [ ] `requirements.txt` unchanged. Training-only deps stay in
  `requirements-train.txt`.
- [ ] `pytest -m 'not slow'` still passes.

## Handoff

Return control to the user for the model-3 (§4) spec expansion. Do NOT proceed
without one — see plan `05-transformer-roadmap.md`.
