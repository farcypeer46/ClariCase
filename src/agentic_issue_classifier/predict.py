"""Run the agentic issue classifier on a split and write metrics + predictions.

Usage:
    # Default (Groq, openai/gpt-oss-120b). Oracle team, 200 rows.
    GROQ_API_KEY=... python -m src.agentic_issue_classifier.predict \
        --split val --rows 200

    # Use team predictions from improved_model_3 as upstream team source
    GROQ_API_KEY=... python -m src.agentic_issue_classifier.predict \
        --split test --rows 500 \
        --team-source docs/metrics/improved_model_3/predictions_test.parquet

    # Switch backend
    AIC_LLM_BACKEND=openai OPENAI_API_KEY=... python -m \
        src.agentic_issue_classifier.predict --split val --rows 100
    AIC_LLM_BACKEND=hf_local python -m \
        src.agentic_issue_classifier.predict --split val --rows 50

Outputs under logs/agentic_issue_classifier/:
    metrics_<split>.json
    predictions_<split>.parquet
    traces_<split>.jsonl        (per-row agent trace: prompts omitted; outputs kept)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.agentic_issue_classifier.config import (
    LLM_BACKEND, LOG_DIR, NUM_FEW_SHOTS, SEED,
)
from src.agentic_issue_classifier.pipeline import AgenticIssuePipeline
from src.agentic_issue_classifier.retriever import build_retriever
from src.utils.config import CACHE


def _load_split_minimal(split: str) -> pd.DataFrame:
    # pd.read_parquet via the default loader materializes every column and
    # OOMs on the wide test parquet. Project only the four we need.
    path = CACHE / "splits_v2" / f"{split}.parquet"
    return pd.read_parquet(
        path,
        columns=["complaint_id", "complaint_text", "team_id", "issue_id"],
    )


def _load_team_source(split: str, source: str | None) -> pd.DataFrame:
    df = _load_split_minimal(split)
    df = df.dropna(subset=["complaint_text", "team_id", "issue_id"])
    if source is None or source == "oracle":
        df["pred_team_id"] = df["team_id"]
        return df.reset_index(drop=True)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"team predictions file not found: {path}")
    preds = pd.read_parquet(path)
    need = {"complaint_id", "pred_team_id"}
    missing = need - set(preds.columns)
    if missing:
        raise KeyError(f"team predictions missing {missing}; got "
                       f"{preds.columns.tolist()}")
    preds = preds[["complaint_id", "pred_team_id"]].astype(
        {"complaint_id": "string", "pred_team_id": "string"})
    df["complaint_id"] = df["complaint_id"].astype("string")
    merged = df.merge(preds, on="complaint_id", how="inner")
    return merged.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--rows", type=int, default=200,
                   help="Subsample N rows (seeded). Use 0 for full split.")
    p.add_argument("--k-shots", type=int, default=NUM_FEW_SHOTS)
    p.add_argument("--verify", action="store_true", default=True)
    p.add_argument("--no-verify", dest="verify", action="store_false")
    p.add_argument("--verify-threshold", type=float, default=0.6)
    p.add_argument("--team-source", default="oracle",
                   help='"oracle" (ground-truth team) or path to a predictions '
                        'parquet with columns [complaint_id, pred_team_id].')
    p.add_argument("--rebuild-retriever", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = _load_team_source(args.split, args.team_source)
    if args.rows and len(df) > args.rows:
        df = df.sample(n=args.rows, random_state=SEED).reset_index(drop=True)
    print(f"[predict] split={args.split} rows={len(df)} "
          f"team_source={args.team_source}", flush=True)

    retriever = build_retriever(force=args.rebuild_retriever)
    pipeline = AgenticIssuePipeline(
        retriever=retriever,
        k_shots=args.k_shots,
        verify=args.verify,
        verify_threshold=args.verify_threshold,
    )

    traces_path = LOG_DIR / f"traces_{args.split}.jsonl"
    traces_path.unlink(missing_ok=True)
    preds, confs, verified_flags, correct_flags = [], [], [], []
    t0 = time.time()
    bar = tqdm(df.itertuples(index=False), total=len(df),
               desc=f"agentic[{LLM_BACKEND}]", unit="row", dynamic_ncols=True)
    with traces_path.open("w", encoding="utf-8") as tf:
        for row in bar:
            res = pipeline.predict(row.complaint_text, row.pred_team_id)
            is_correct = res.issue_id == str(row.issue_id)
            preds.append(res.issue_id)
            confs.append(res.confidence)
            verified_flags.append(res.verified)
            correct_flags.append(is_correct)
            tf.write(json.dumps({
                "complaint_id": str(row.complaint_id),
                "team_id": str(row.team_id),
                "pred_team_id": str(row.pred_team_id),
                "true_issue_id": str(row.issue_id),
                "pred_issue_id": res.issue_id,
                "confidence": res.confidence,
                "verified": res.verified,
                "correct": is_correct,
                "classifier_raw": res.classifier.get("raw"),
                "verifier_raw": (res.verifier or {}).get("raw"),
            }) + "\n")
            n = len(preds)
            bar.set_postfix({
                "acc": f"{sum(correct_flags) / n:.3f}",
                "unk": f"{sum(p == 'UNK' for p in preds) / n:.3f}",
                "verified": f"{sum(verified_flags) / n:.2f}",
            })
    bar.close()
    wall = time.time() - t0

    y_true = df["issue_id"].astype(str).tolist()
    y_pred = preds
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    unk_rate = sum(p == "UNK" for p in y_pred) / len(y_pred)
    team_match_rate = float((df["team_id"] == df["pred_team_id"]).mean())
    verified_rate = sum(verified_flags) / len(verified_flags)

    metrics = {
        "split": args.split,
        "backend": LLM_BACKEND,
        "rows": len(df),
        "team_source": args.team_source,
        "k_shots": args.k_shots,
        "verify": args.verify,
        "verify_threshold": args.verify_threshold,
        "team_match_rate": round(team_match_rate, 4),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro), 4),
        "weighted_f1": round(float(weighted), 4),
        "unk_rate": round(unk_rate, 4),
        "verified_rate": round(verified_rate, 4),
        "wall_clock_sec": round(wall, 1),
        "throughput_rows_per_sec": round(len(df) / wall, 2),
    }
    (LOG_DIR / f"metrics_{args.split}.json").write_text(
        json.dumps(metrics, indent=2))

    out_df = pd.DataFrame({
        "complaint_id": df["complaint_id"].astype(str),
        "team_id": df["team_id"].astype(str),
        "pred_team_id": df["pred_team_id"].astype(str),
        "true_issue_id": y_true,
        "pred_issue_id": y_pred,
        "confidence": confs,
        "verified": verified_flags,
    })
    out_df.to_parquet(LOG_DIR / f"predictions_{args.split}.parquet",
                      index=False)

    summary_lines = [
        f"Agentic Issue Classifier — {args.split} split",
        f"backend={LLM_BACKEND}  k_shots={args.k_shots}  "
        f"verify={args.verify} (thr={args.verify_threshold})",
        f"team_source={args.team_source}",
        f"rows={len(df)}  wall={wall:.1f}s  "
        f"throughput={len(df) / wall:.2f} rows/s",
        "",
        f"accuracy        {acc:.4f}",
        f"macro_f1        {macro:.4f}",
        f"weighted_f1     {weighted:.4f}",
        f"unk_rate        {unk_rate:.4f}",
        f"verified_rate   {verified_rate:.2%}",
        f"team_match_rate {team_match_rate:.4f}",
    ]
    summary = "\n".join(summary_lines)
    (LOG_DIR / f"summary_{args.split}.txt").write_text(summary,
                                                       encoding="utf-8")
    print("\n" + summary)
    print(f"\n[predict] wrote metrics + predictions + traces + summary "
          f"to {LOG_DIR}", flush=True)


if __name__ == "__main__":
    main()
