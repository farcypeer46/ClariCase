"""Importable + CLI prediction for improved_model_1.

Output is a superset of src.baseline_model_2.predict.predict().
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.improved_model_1.config import ARTIFACT_DIR
from src.utils.config import TEAM_MAPPING


def load_model(path: Path = ARTIFACT_DIR / "model.joblib"):
    return joblib.load(path)


def _team_names() -> dict[str, str]:
    df = pd.read_csv(TEAM_MAPPING)
    return dict(zip(df["team_id"], df["team_name"]))


def predict(text: str, top_k: int = 3, model=None) -> dict:
    bundle = model if model is not None else load_model()
    vec = bundle["vectorizer"]
    hsvm = bundle["model"]
    team_classes = np.asarray(bundle["team_classes"])
    issue_labels = bundle.get("issue_labels", {}) or {}
    global_thr = bundle.get("global_threshold")
    model_name = bundle.get("model_name", "improved_model_1")

    team_lookup = _team_names()

    X = vec.transform([text])
    proba = hsvm.team_proba(X)[0]
    order = np.argsort(proba)[::-1][:top_k]
    predicted_team_id = str(team_classes[order[0]])
    confidence = float(proba[order[0]])

    team_top = [{
        "team_id": str(team_classes[j]),
        "team_name": team_lookup.get(str(team_classes[j]), ""),
        "confidence": float(proba[j]),
    } for j in order]

    issue_top_raw = hsvm.predict_issue_topk(X, teams=np.array([predicted_team_id]),
                                            k=top_k)[0]
    issue_top_k = [{
        "issue_id": iid,
        "issue_label": issue_labels.get(iid, ""),
        "confidence": conf,
    } for iid, conf in issue_top_raw]

    # Served routing uses the single global threshold: the evaluated headline
    # operating point (61.5% auto-routed at 94.8% precision on test). Per-team
    # thresholds stay in the bundle for analysis but do not drive routing.
    route = ("auto" if global_thr is not None and confidence >= global_thr
             else "review")

    return {
        "predicted_team_id": predicted_team_id,
        "predicted_team_name": team_lookup.get(predicted_team_id, ""),
        "confidence": confidence,
        "top_k": team_top,
        "issue_top_k": issue_top_k,
        "route": route,
        "model_name": model_name,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    text = args.text or sys.stdin.read()
    if not text.strip():
        raise SystemExit("no complaint text provided")

    print(json.dumps(predict(text, top_k=args.top_k), indent=2))
