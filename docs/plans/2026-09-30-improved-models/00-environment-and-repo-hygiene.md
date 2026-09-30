# 00 — Environment & Repo Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare the repo and Python environment so every downstream sub-plan
can run: enable Git LFS, delete a stray scratch file, split training-only
dependencies out of `requirements.txt`, and (optionally) install a CUDA-enabled
PyTorch build.

**Architecture:** No production code changes. Repo-level edits only:
`.gitignore`, `requirements.txt`, new `requirements-train.txt`, delete
`data/processed/splits/w2v_corpus.csv`. Verify the working tree is clean of
line-ending noise before branching.

**Tech Stack:** git, git-lfs, pip, Python 3.14, PyTorch (CUDA).

---

## Files touched

- Modify: `F:/NLP_MSML641/.gitignore`
- Modify: `F:/NLP_MSML641/requirements.txt` (only to add `supabase` if missing;
  see Task 4)
- Create: `F:/NLP_MSML641/requirements-train.txt`
- Delete: `F:/NLP_MSML641/data/processed/splits/w2v_corpus.csv`

---

### Task 1: Verify branch state and enable Git LFS

- [ ] **Step 1: Confirm current branch and clean status**

Run:
```bash
cd F:/NLP_MSML641 && git status --short && git branch --show-current
```
Expected: current branch prints (may be `baseline_model_2` or already updated);
only expected modifications are line-ending / CRLF noise plus the two LFS zips.

- [ ] **Step 2: Show that "modified" files are only CRLF noise**

Run:
```bash
cd F:/NLP_MSML641 && git diff --ignore-all-space --stat
```
Expected: empty output (or only the two LFS zips listed). If real diffs appear,
STOP and report to user before touching anything else.

- [ ] **Step 3: Install Git LFS locally (one-time)**

Run:
```bash
git lfs install
```
Expected: `Git LFS initialized.`

- [ ] **Step 4: Verify LFS-tracked files re-resolve**

Run:
```bash
cd F:/NLP_MSML641 && git lfs ls-files
```
Expected: at least
`data/complaint_router_11_teams_2024_2025.zip` and
`data/complaint_router_2026_holdout.zip` listed.

- [ ] **Step 5: Commit-free checkpoint**

No commit — this task only touches local git config.

---

### Task 2: Delete stray scratch file

- [ ] **Step 1: Confirm the file exists and is not tracked**

Run:
```bash
cd F:/NLP_MSML641 && ls data/processed/splits/w2v_corpus.csv && git ls-files --error-unmatch data/processed/splits/w2v_corpus.csv 2>&1
```
Expected: file listed; second command exits non-zero with
`error: pathspec 'data/processed/splits/w2v_corpus.csv' did not match any file(s) known to git`.

- [ ] **Step 2: Delete it**

Run:
```bash
rm F:/NLP_MSML641/data/processed/splits/w2v_corpus.csv
```

- [ ] **Step 3: Verify remainder of splits/ is untouched**

Run:
```bash
ls F:/NLP_MSML641/data/processed/splits/
```
Expected: no `w2v_corpus.csv`; other CSVs (owned by baseline 2) still present.

- [ ] **Step 4: No commit needed** — the file was untracked.

---

### Task 3: Update `.gitignore` for future caches and checkpoints

- [ ] **Step 1: Read current `.gitignore`**

Read: `F:/NLP_MSML641/.gitignore`.

- [ ] **Step 2: Append new rules**

Edit `F:/NLP_MSML641/.gitignore`, adding a new section at the bottom:
```
# ---------------------------------------------------------------- improved models
data/cache/
*.npy
*.kv
**/checkpoints/
```

- [ ] **Step 3: Verify current pattern for `.joblib` is not blanket**

Run:
```bash
grep -n joblib F:/NLP_MSML641/.gitignore
```
Expected: line contains `src/baseline_model_1/artifacts/*.joblib` and NOT a
blanket `*.joblib` (baseline 2's `model.joblib` under `docs/metrics/` must
remain tracked). If a blanket rule appears, remove it.

- [ ] **Step 4: Stage and commit**

Run:
```bash
cd F:/NLP_MSML641 && git add .gitignore
git commit -m "chore: ignore improved-model caches and checkpoints"
```

---

### Task 4: Create `requirements-train.txt`

- [ ] **Step 1: Create the file**

Write `F:/NLP_MSML641/requirements-train.txt`:
```
# Training-only dependencies. NOT installed by Streamlit Community Cloud.
# Install locally with: pip install -r requirements-train.txt
#
# torch: install the CUDA build separately, e.g.
#   python -m pip install --force-reinstall --no-deps torch \
#       --index-url https://download.pytorch.org/whl/cu128
# The unpinned entry below is only for CPU fallback / CI.
torch
gensim
sentence-transformers
transformers
datasets
accelerate
peft
captum
```

- [ ] **Step 2: Check whether `supabase` is missing from `requirements.txt`**

Run:
```bash
grep -n '^supabase' F:/NLP_MSML641/requirements.txt || echo MISSING
```
Expected: prints `MISSING`. If present, skip Step 3.

- [ ] **Step 3: Flag `supabase` gap to the user (do NOT auto-add)**

Print:
> "`requirements.txt` does not list `supabase`, but the deployed app imports it.
> Flag this so the user can decide whether adding it belongs in this hygiene
> commit or in the app-integration plan."

STOP for user decision before editing `requirements.txt`. Once approved:
```bash
# Only if user approves:
echo "supabase" >> F:/NLP_MSML641/requirements.txt
```

- [ ] **Step 4: Stage and commit**

Run:
```bash
cd F:/NLP_MSML641 && git add requirements-train.txt
# add requirements.txt only if it was actually edited in Step 3
git commit -m "chore: split training-only deps into requirements-train.txt"
```

---

### Task 5 (OPTIONAL — only if GPU work will run locally): Install CUDA PyTorch

Skip this task entirely if all subsequent work will run scikit-learn only (§2)
or on a machine that does not need GPU. Model 2's `sbert` variant and every
transformer model (§4) require GPU torch.

- [ ] **Step 1: Confirm NVIDIA driver + CUDA version**

Run:
```bash
nvidia-smi
```
Expected: driver + CUDA version printed. If command not found, STOP — this
machine has no NVIDIA GPU, and Task 5 is not applicable.

- [ ] **Step 2: Confirm current torch is CPU-only**

Run:
```bash
F:/NLP_MSML641/.venv/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Expected: version ends in `+cpu`, `cuda.is_available()` prints `False`.

- [ ] **Step 3: Reinstall torch as a CUDA build**

Pick the URL matching the CUDA reported in Step 1 (e.g. `cu128` for CUDA 12.8):
```bash
F:/NLP_MSML641/.venv/Scripts/python -m pip install --force-reinstall --no-deps \
    torch --index-url https://download.pytorch.org/whl/cu128
```
`--no-deps` keeps pip from touching other packages (scikit-learn stays at 1.9.1).

- [ ] **Step 4: Verify CUDA torch works and scikit-learn is unchanged**

Run:
```bash
F:/NLP_MSML641/.venv/Scripts/python -c "import torch, sklearn; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', sklearn.__version__)"
```
Expected: torch version ends in `+cuXXX`, `cuda.is_available()` prints `True`,
GPU name printed, and sklearn still reports `1.9.1`.

- [ ] **Step 5: No commit** — venv is not tracked. Record the exact `pip
  install` command in `docs/improved_model_1.md` when that doc is written.

---

### Task 6: Create the working branch

- [ ] **Step 1: Ensure clean tree before branching**

Run:
```bash
cd F:/NLP_MSML641 && git status --short
```
Expected: empty (or only the LFS zip lines if LFS is still resolving).

- [ ] **Step 2: Create branch `improved_model_1` from current HEAD**

Run:
```bash
cd F:/NLP_MSML641 && git checkout -b improved_model_1
```
Expected: `Switched to a new branch 'improved_model_1'`.

- [ ] **Step 3: Verify**

Run:
```bash
git branch --show-current
```
Expected: `improved_model_1`.

---

## Self-review checklist

- [ ] All spec §0 items covered: rule 0.8 (`.gitignore`), rule 0.10 (delete
  `w2v_corpus.csv`), rule 0.11 (`requirements-train.txt`), §0.1 (CUDA torch).
- [ ] `requirements.txt` untouched (except optional `supabase` add flagged
  through the user).
- [ ] `sklearn` version still 1.9.1 (verified in Task 5 Step 4).
- [ ] `src/baseline_model_*/` and `data/processed/splits/` (other than
  `w2v_corpus.csv`) untouched.
- [ ] Branch `improved_model_1` exists locally, based on current HEAD.

## Handoff

Next plan: `01-shared-foundation-utils.md`.
