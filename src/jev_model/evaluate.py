"""Evaluate Jev as a zero-shot team classifier against Improved Model 1.

Both models see the same sampled complaints, PII-masked exactly as the app
masks them, and are scored with the same harness: the routing threshold is
fit on the validation sample and frozen for the test sample.

    python -m src.jev_model.evaluate --mock          # offline, no key, no cost
    python -m src.jev_model.evaluate --probe         # one real request, raw reply
    python -m src.jev_model.evaluate                 # prints cost estimate, stops
    python -m src.jev_model.evaluate --yes           # real run (2,000 + 2,000)

Answers are cached per complaint in data/cache/jev_model/, so an interrupted
run resumes without paying for completed requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.app.pii import mask
from src.jev_model.client import (JevClient, JevError, MockJevClient,
                                  load_api_key)
from src.jev_model.config import (ARTIFACT_DIR, CACHE_DIR, CONCURRENCY, MODEL,
                                  N_TEST_DEFAULT, N_VAL_DEFAULT,
                                  PRICE_PER_M_INPUT_TOKENS, SEED, TEAM_IDS,
                                  team_question)
from src.utils.labels import team_names
from src.utils.metrics import evaluate
from src.utils.routing import north_star_summary

PROBE_TEXT = ("A debt collector keeps calling me about a loan I already paid "
              "off two years ago and threatened to report it to the credit "
              "bureaus.")


# ---------------------------------------------------------------- data
def sample_split(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Random sample at the real team mix, with the masked model input."""
    out = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    out["model_text"] = out["complaint_text"].astype(str).map(mask)
    out["text_sha256"] = out["model_text"].map(
        lambda t: hashlib.sha256(t.encode()).hexdigest())
    return out


def _question_tokens() -> int:
    return len(json.dumps(team_question())) // 4


def estimate_cost(texts: list[str]) -> tuple[int, float]:
    tokens = sum(len(t) // 4 + _question_tokens() for t in texts)
    return tokens, tokens / 1e6 * PRICE_PER_M_INPUT_TOKENS


# ---------------------------------------------------------------- cache
def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {r["complaint_id"]: r for r in rows}


def _pending(df: pd.DataFrame, cache: dict[str, dict]) -> pd.DataFrame:
    hit = df.apply(lambda r: (str(r["complaint_id"]) in cache and
                              cache[str(r["complaint_id"])]["text_sha256"]
                              == r["text_sha256"]), axis=1)
    return df[~hit]


# ---------------------------------------------------------------- classify
def classify_frame(client, df: pd.DataFrame, cache_path: Path,
                   label: str = "") -> np.ndarray:
    """Probability matrix (rows x TEAM_IDS) for df, reusing cached answers."""
    cache = load_cache(cache_path)
    todo = _pending(df, cache)
    if len(todo):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock = threading.Lock()
        done = 0
        with open(cache_path, "a") as f, ThreadPoolExecutor(CONCURRENCY) as pool:
            futures = {pool.submit(client.classify_team, r.model_text): r
                       for r in todo.itertuples()}
            for fut in as_completed(futures):
                r = futures[fut]
                ans = fut.result()          # JevError stops the run; cache kept
                row = {"complaint_id": str(r.complaint_id),
                       "text_sha256": r.text_sha256,
                       "probabilities": ans.probabilities, "model": ans.model,
                       "input_tokens": ans.input_tokens}
                with lock:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    cache[row["complaint_id"]] = row
                    done += 1
                    if done % 100 == 0 or done == len(todo):
                        print(f"  [{label}] {done:,}/{len(todo):,} classified")
    return np.array([[cache[str(c)]["probabilities"][t] for t in TEAM_IDS]
                     for c in df["complaint_id"]])


def improved_model_1_proba(texts: list[str]) -> np.ndarray:
    """Improved Model 1's calibrated team probabilities, in TEAM_IDS order."""
    from src.improved_model_1.predict import load_model
    bundle = load_model()
    proba = bundle["model"].team_proba(bundle["vectorizer"].transform(texts))
    order = [list(bundle["team_classes"]).index(t) for t in TEAM_IDS]
    return proba[:, order]


# ---------------------------------------------------------------- scoring
def score(val: pd.DataFrame, val_proba: np.ndarray,
          test: pd.DataFrame, test_proba: np.ndarray) -> dict:
    teams = np.array(TEAM_IDS)
    out = {}
    for name, df, proba in (("val", val, val_proba), ("test", test, test_proba)):
        out[name] = evaluate(df["team_id"].to_numpy(), proba, teams,
                             bootstrap=False)["team"]
    v_pred, t_pred = teams[val_proba.argmax(1)], teams[test_proba.argmax(1)]
    summary, curves, per_team = north_star_summary(
        val_proba.max(1), v_pred == val["team_id"].to_numpy(),
        test_proba.max(1), t_pred == test["team_id"].to_numpy(),
        test["team_id"].to_numpy(), t_pred, team_names)
    summary["protocol"] = ("same random samples for every model, PII-masked; "
                           "threshold fit on the validation sample, frozen for "
                           "test; Jev probabilities used as returned")
    out["north_star"] = summary
    out["_curves"], out["_per_team"] = curves, per_team
    return out


def _json(obj) -> str:
    return json.dumps(obj, indent=2,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))


def write_artifacts(out_dir: Path, results: dict, run_info: dict,
                    test: pd.DataFrame, probas: dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    teams = np.array(TEAM_IDS)
    metrics = {}
    for model, res in results.items():
        metrics[model] = {k: res[k] for k in ("val", "test")}
        (out_dir / f"north_star_{model}.json").write_text(_json(res["north_star"]))
        res["_curves"].to_csv(out_dir / f"coverage_curve_{model}.csv", index=False)
        if res["_per_team"] is not None:
            res["_per_team"].to_csv(out_dir / f"per_team_routing_{model}.csv",
                                    index=False)
    (out_dir / "metrics.json").write_text(_json(metrics))
    (out_dir / "run_info.json").write_text(_json(run_info))
    preds = test[["complaint_id", "team_id"]].rename(columns={"team_id": "actual"})
    for model, proba in probas.items():
        preds[f"{model}_pred"] = teams[proba.argmax(1)]
        preds[f"{model}_conf"] = proba.max(1).round(4)
    preds.to_csv(out_dir / "predictions_test.csv", index=False)


# ---------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-val", type=int, default=N_VAL_DEFAULT)
    p.add_argument("--n-test", type=int, default=N_TEST_DEFAULT)
    p.add_argument("--mock", action="store_true",
                   help="offline mock client: no key, no cost, meaningless scores")
    p.add_argument("--yes", action="store_true",
                   help="confirm the estimated cost and make real requests")
    p.add_argument("--probe", action="store_true",
                   help="send one request and print the raw response, then exit")
    p.add_argument("--no-baseline", action="store_true",
                   help="skip the Improved Model 1 comparison")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    client = MockJevClient() if args.mock else None
    if client is None:
        try:
            client = JevClient(load_api_key())
        except JevError as exc:
            print(f"[jev] {exc}\n      Use --mock to test the pipeline offline.")
            return 1

    if args.probe:
        body = client.raw(mask(PROBE_TEXT)) if hasattr(client, "raw") else {}
        print(_json(body))
        return 0

    from src.utils.data import load_split
    val = sample_split(load_split("val"), args.n_val)
    test = sample_split(load_split("test"), args.n_test, seed=SEED + 1)

    tag = "mock" if args.mock else MODEL
    cache_path = CACHE_DIR / f"{tag}.jsonl"
    pending = pd.concat([_pending(val, load_cache(cache_path)),
                         _pending(test, load_cache(cache_path))])
    tokens, cost = estimate_cost(pending["model_text"].tolist())
    print(f"[jev] model={client.model}  val={len(val):,}  test={len(test):,}  "
          f"new requests={len(pending):,}  est. input tokens={tokens:,}  "
          f"est. cost=${cost:.2f}" + ("  (mock: free)" if args.mock else ""))
    if not args.mock and len(pending) and not args.yes:
        print("[jev] Nothing sent. Re-run with --yes to make these requests.")
        return 0

    val_jev = classify_frame(client, val, cache_path, "val")
    test_jev = classify_frame(client, test, cache_path, "test")
    results = {"jev": score(val, val_jev, test, test_jev)}
    probas = {"jev": test_jev}
    if not args.no_baseline:
        print("[jev] scoring Improved Model 1 on the same samples")
        val_im1 = improved_model_1_proba(val["model_text"].tolist())
        test_im1 = improved_model_1_proba(test["model_text"].tolist())
        results["improved_model_1"] = score(val, val_im1, test, test_im1)
        probas["improved_model_1"] = test_im1

    cache = load_cache(cache_path)
    used = [cache[str(c)] for c in pd.concat([val, test])["complaint_id"]]
    run_info = {
        "mock": args.mock, "requested_model": client.model,
        "answering_models": sorted({u["model"] for u in used}),
        "n_val": len(val), "n_test": len(test), "seed": SEED,
        "input_tokens": int(sum(u["input_tokens"] for u in used)),
        "est_cost_usd_at_published_price": round(
            sum(u["input_tokens"] for u in used) / 1e6 * PRICE_PER_M_INPUT_TOKENS, 4),
        "pii_masked": True, "run_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
    }
    # Mock output stays in the gitignored cache so it cannot pass for results.
    out_dir = CACHE_DIR / "mock_artifacts" if args.mock else ARTIFACT_DIR
    write_artifacts(out_dir, results, run_info, test, probas)

    print("\n[results] test sample" + ("  (MOCK: scores are meaningless)"
                                       if args.mock else ""))
    for model, res in results.items():
        ns = res["north_star"]
        print(f"  {model:<17} macro-F1 {res['test']['macro_f1']:.3f}  "
              f"accuracy {res['test']['accuracy']:.3f}  "
              f"auto-routed@95% {ns['value'] if ns['value'] is not None else 'n/a'}")
    print(f"[write] {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
