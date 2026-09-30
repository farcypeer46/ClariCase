# 05 — Transformer Models 3 / 4 / 5 Roadmap (STUB — DO NOT EXECUTE)

**Status:** intentional stub. The source spec (`docs/implementation_plan_improved_models.md`
§4) explicitly says these folders "stay **empty** until their specs are
written." This file is a placeholder so no one starts a transformer run without
a dedicated, task-level plan.

**When to expand this file:** the user asks for model 3 (or 4 or 5) *by name*.
At that point, write a full task-level plan matching the style of plans 02 and
03 in this folder, save it as `06-improved-model-3-bert-base.md` (or similar),
and only then create files under `src/improved_model_3/`.

---

## Common contract these plans MUST inherit (from §4)

Any future transformer plan must include:

1. **Data:** reuse `src.utils.data.load_split` and `load_holdout_2026()` — do
   **not** rebuild splits. `train_cap_issue` is the main training set.
2. **Preprocessing:** raw text minus HTML unescape and whitespace
   normalization. `XXXX` redaction runs, `{$…}`, and `XX/XX/XXXX` map to
   `[REDACTED]`, `[AMOUNT]`, `[DATE]` special tokens **added to the tokenizer**
   before training. Do NOT reuse `clean_text` here (it deletes those tokens).
3. **Truncation:** `max_len=512` (BERT/DeBERTa) or `1024` (ModernBERT).
   Head+tail truncation, not head-only.
4. **Seeds:** `transformers.set_seed(42)` in every entry point.
5. **Device:** `src.utils.device.get_device()`. Never hard-code `"cuda"`.
   AMP dtype from `src.utils.device.amp_dtype()`
   (`bf16` on supported hardware, else `fp16`). Batch size from
   `suggested_batch_size(vram, task)`.
6. **Class weights:** class-weighted cross-entropy; weights computed on train
   only.
7. **Calibration:** temperature scaling fit on validation. Then reuse
   `src.utils.routing` for global and per-team thresholds.
8. **Checkpoints:** `save_total_limit=1`, `load_best_model_at_end=True`,
   `metric_for_best_model = "team_macro_f1"`. Store checkpoints under
   `data/cache/improved_model_N/checkpoints/` (gitignored).
9. **Logging to `run_info.json`:** `gpu_info()`, peak VRAM
   (`torch.cuda.max_memory_allocated()`), wall-clock training time, inference
   throughput, model card / HF hash, epoch count, LR schedule.
10. **Artifacts contract:** every folder writes the same files as
    `improved_model_1/` — `metrics.json`, `north_star.json`,
    `per_team_routing.csv`, `predictions_test.csv`, plus a `run_info.json`.
11. **Registry:** append rows for train, val, test, and holdout_2026 via
    `src.utils.registry.log_result`.
12. **Doc:** `docs/improved_model_N.md` in the same structure as
    `docs/improved_model_1.md`.
13. **Serving:** transformer models are **NOT** servable from Streamlit
    Community Cloud (no GPU, ~1 GB RAM). Options at that point: ONNX / int8
    quantization, distillation into a small encoder, or an external inference
    endpoint that keeps the `route()` contract in `src/utils/serving.py`. Do
    not touch `streamlit_app.py` here.

---

## Per-model open questions to resolve before writing the plan

### Model 3 — Fine-tuned BERT-base
- Team head only, or team head then issue head sequentially?
- Which layers to unfreeze in each phase, at what learning rates?
  (Spec suggests heads → top ⅓ → all with LLRD=0.85.)
- Evaluation cadence on a 20k val subset — how is the subset sampled to stay
  representative of the real team mix?

### Model 4 — DeBERTa-v3-base and ModernBERT-base (hierarchical multi-task)
- Loss weighting: `CE_team + 0.5 · CE_issue`? Or tuned?
- LoRA (r=16) as an ablation vs full fine-tune — decide gate first (does full
  fine-tuning fit on target GPU?).
- Precision: bf16 (DeBERTa hates fp16), fp32 fallback path if unstable.

### Model 5 — DAPT + calibration/conformal + ensemble
- Continued MLM on 774,412 train texts, 1 epoch, 30% masking. Where does the
  DAPT checkpoint live? (`data/cache/improved_model_5/dapt/`, checkpointed
  every N steps for resume.)
- Ensemble weight tuning: fixed 0.5/0.5 vs learned on val?
- Conformal prediction sets: which coverage target (90%)? Does the app UI need
  to render sets, or only single-team routing?

### Later
- PII scrubbing (`src/utils/pii.py` with Presidio + regex) applied in the app
  **before storing or showing** a complaint.
- LLM adjudicator for the "review" queue.
- ONNX/int8/distilled export for cheap serving.

---

## Bootstrap steps when a real plan is written

1. Create the branch (`improved_model_3`, etc.).
2. Add `src/improved_model_N/__init__.py` (was empty until this point).
3. Add `pyproject.toml` or `requirements-train.txt` entries for any new libs
   (peft, accelerate).
4. Follow the same file-structure conventions as `src/improved_model_1/`:
   `config.py`, `features.py` (or `tokenizer.py`), `model.py`, `train.py`,
   `predict.py`, plus `docs/improved_model_N.md`.

---

## Explicit non-goals of this stub

- No code changes.
- No new folders under `src/`.
- No commits when this file is opened.
- The empty `src/improved_model_3/`, `src/improved_model_4/`,
  `src/improved_model_5/` folders exist as placeholders per the source spec —
  **leave them empty**.
