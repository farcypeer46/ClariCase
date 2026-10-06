"""Train and evaluate improved_model_1 (hierarchical TF-IDF + LinearSVC).

Usage:
    python -m src.improved_model_1.train [--no-char] [--quiet]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from src.improved_model_1.config import (
    ARTIFACT_DIR, C_ISSUE_GRID, C_TEAM_GRID, CHAR_MAX_FEATURES,
    MAX_ARTIFACT_MB, MIN_ISSUE_ROWS_ISOTONIC, USE_CHAR_DEFAULT,
    WORD_MAX_FEATURES,
)
from src.improved_model_1.features import build_vectorizer
from src.improved_model_1.model import HierarchicalSVM
from src.utils import config as C
from src.utils.data import load_holdout_2026, load_split
from src.utils.device import gpu_info
from src.utils.labels import issue_labels, issue_to_team
from src.utils.registry import log_result
from src.utils.routing import (
    apply_per_team_thresholds, expected_calibration_error, north_star_summary,
    per_team_thresholds,
)
from src.utils.text import clean_text


# ------------------------- loading and tuning --------------------------------

def _load_all_splits() -> dict[str, pd.DataFrame]:
    return {
        "train": load_split("train_cap_issue"),
        "val": load_split("val"),
        "test": load_split("test"),
        "holdout_2026": load_holdout_2026(),
    }


def _tune(vec, train: pd.DataFrame, val: pd.DataFrame,
          verbose: bool) -> tuple[float, float, dict]:
    """Grid-search C_team then C_issue on validation."""
    X_train = vec.fit_transform(train[C.TEXT_COL].astype(str))
    X_val = vec.transform(val[C.TEXT_COL].astype(str))
    y_train_t = train[C.TEAM_COL].to_numpy()
    y_val_t = val[C.TEAM_COL].to_numpy()
    y_train_i = train[C.ISSUE_COL].to_numpy()
    y_val_i = val[C.ISSUE_COL].to_numpy()

    team_grid = []
    for C_t in C_TEAM_GRID:
        if verbose:
            print(f"  [tune] C_team={C_t} ...", flush=True)
        m = HierarchicalSVM(C_team=C_t, C_issue=1.0).fit(
            X_train, y_train_t, y_train_i)
        pred = m.predict_team(X_val)
        f1 = float(f1_score(y_val_t, pred, average="macro", zero_division=0))
        team_grid.append({"C_team": C_t, "val_team_macro_f1": f1})
        if verbose:
            print(f"      val team macro-F1 = {f1:.4f}")
    best_team = max(team_grid, key=lambda r: r["val_team_macro_f1"])

    issue_grid = []
    for C_i in C_ISSUE_GRID:
        if verbose:
            print(f"  [tune] C_team={best_team['C_team']} "
                  f"C_issue={C_i} (oracle team) ...", flush=True)
        m = HierarchicalSVM(C_team=best_team["C_team"], C_issue=C_i).fit(
            X_train, y_train_t, y_train_i)
        top = m.predict_issue_topk(X_val, teams=y_val_t, k=1)
        pred_i = np.array([r[0][0] for r in top])
        f1 = float(f1_score(y_val_i, pred_i, average="macro", zero_division=0))
        issue_grid.append({"C_issue": C_i, "val_issue_macro_f1_oracle": f1})
        if verbose:
            print(f"      val issue macro-F1 (oracle) = {f1:.4f}")
    best_issue = max(issue_grid, key=lambda r: r["val_issue_macro_f1_oracle"])

    return best_team["C_team"], best_issue["C_issue"], {
        "team_grid": team_grid, "issue_grid": issue_grid,
        "best_team": best_team, "best_issue": best_issue,
    }


# ---------------------- calibration + routing --------------------------------

def _calibrate_and_route(model: HierarchicalSVM, vec, val: pd.DataFrame,
                         test: pd.DataFrame, names: dict[str, str],
                         verbose: bool) -> dict:
    if verbose:
        print("  [calibrate] isotonic on validation", flush=True)
    X_val = vec.transform(val[C.TEXT_COL].astype(str))
    model.fit_team_calibrator(X_val, val[C.TEAM_COL].to_numpy())
    model.fit_issue_calibrators(X_val, val[C.TEAM_COL].to_numpy(),
                                val[C.ISSUE_COL].to_numpy(),
                                min_rows=MIN_ISSUE_ROWS_ISOTONIC)

    X_test = vec.transform(test[C.TEXT_COL].astype(str))
    val_proba = model.team_proba(X_val)
    test_proba = model.team_proba(X_test)
    team_classes = model.calibrated_team_.classes_
    val_pred = team_classes[val_proba.argmax(1)]
    val_conf = val_proba.max(1)
    test_pred = team_classes[test_proba.argmax(1)]
    test_conf = test_proba.max(1)

    y_val = val[C.TEAM_COL].to_numpy()
    y_test = test[C.TEAM_COL].to_numpy()
    val_correct = val_pred == y_val
    test_correct = test_pred == y_test

    if verbose:
        print("  [route] computing north-star + per-team thresholds",
              flush=True)
    ns, curves, teams = north_star_summary(
        val_conf, val_correct, test_conf, test_correct,
        y_test, test_pred, names)
    per_team_thr = per_team_thresholds(val_conf, val_pred, y_val, target=0.95)
    routing_ptt = apply_per_team_thresholds(
        y_test, test_pred, test_conf, per_team_thr, names=names)

    return {
        "ns": ns, "curves": curves, "teams": teams,
        "per_team_thresholds": per_team_thr,
        "routing_ptt": routing_ptt,
        "val_pred": val_pred, "val_conf": val_conf,
        "test_pred": test_pred, "test_conf": test_conf,
        "test_proba": test_proba, "team_classes": team_classes,
        "X_test": X_test,
    }


# --------------------------- evaluate + write --------------------------------

def _issue_top_k(model: HierarchicalSVM, X, teams: np.ndarray, k: int = 3
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (top1, top2, top3) issue IDs per row (teams index chooses the
    issue head)."""
    top = model.predict_issue_topk(X, teams=teams, k=k)
    top1 = np.array([r[0][0] if r else "" for r in top])
    top2 = np.array([r[1][0] if len(r) > 1 else "" for r in top])
    top3 = np.array([r[2][0] if len(r) > 2 else "" for r in top])
    return top1, top2, top3


def _write_artifacts(model: HierarchicalSVM, vec, train, val, test,
                     holdout, routed: dict, tuning: dict, args,
                     wall_clock_sec: float, verbose: bool) -> None:
    output = ARTIFACT_DIR
    output.mkdir(parents=True, exist_ok=True)

    ns = routed["ns"]
    y_test = test[C.TEAM_COL].to_numpy()
    y_test_i = test[C.ISSUE_COL].to_numpy()
    test_pred = routed["test_pred"]
    test_conf = routed["test_conf"]
    X_test = routed["X_test"]
    team_classes = routed["team_classes"]

    # -- team metrics
    val_pred = routed["val_pred"]
    val_metrics = {
        "accuracy": float((val_pred == val[C.TEAM_COL].to_numpy()).mean()),
        "macro_f1": float(f1_score(val[C.TEAM_COL], val_pred,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(val[C.TEAM_COL], val_pred,
                                      average="weighted", zero_division=0)),
    }
    test_metrics = {
        "accuracy": float((test_pred == y_test).mean()),
        "macro_f1": float(f1_score(y_test, test_pred,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, test_pred,
                                      average="weighted", zero_division=0)),
    }

    dummy = DummyClassifier(strategy="most_frequent").fit(
        train[[C.TEXT_COL]], train[C.TEAM_COL])
    dpred_v = dummy.predict(val[[C.TEXT_COL]])
    dpred_t = dummy.predict(test[[C.TEXT_COL]])
    dummy_metrics = {
        "val": {
            "accuracy": float((dpred_v == val[C.TEAM_COL].to_numpy()).mean()),
            "macro_f1": float(f1_score(val[C.TEAM_COL], dpred_v,
                                       average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(val[C.TEAM_COL], dpred_v,
                                          average="weighted", zero_division=0)),
        },
        "test": {
            "accuracy": float((dpred_t == y_test).mean()),
            "macro_f1": float(f1_score(y_test, dpred_t,
                                       average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, dpred_t,
                                          average="weighted", zero_division=0)),
        },
    }

    labels = sorted(set(train[C.TEAM_COL]) | set(y_test))
    report = classification_report(y_test, test_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    cmat = confusion_matrix(y_test, test_pred, labels=labels)

    # -- issue metrics (hierarchical: use predicted team)
    i_top1, i_top2, i_top3 = _issue_top_k(model, X_test, test_pred, k=3)
    i_oracle_top1, _, _ = _issue_top_k(model, X_test, y_test, k=3)
    issue_metrics = {
        "accuracy_predicted_team": float((i_top1 == y_test_i).mean()),
        "macro_f1_predicted_team": float(f1_score(
            y_test_i, i_top1, average="macro", zero_division=0)),
        "accuracy_oracle_team": float((i_oracle_top1 == y_test_i).mean()),
        "macro_f1_oracle_team": float(f1_score(
            y_test_i, i_oracle_top1, average="macro", zero_division=0)),
    }
    issue_labels_expected = sorted(set(y_test_i) | set(i_top1))
    report_issue = classification_report(
        y_test_i, i_top1, labels=issue_labels_expected,
        output_dict=True, zero_division=0)

    # -- joint and hard pairs
    joint_exact = float(((test_pred == y_test) & (i_top1 == y_test_i)).mean())
    def _recall(tid):
        own = y_test == tid
        return float((test_pred[own] == tid).mean()) if own.any() else None
    def _share_as(tid, other):
        own = y_test == tid
        if not own.any():
            return None
        return float((test_pred[own] == other).mean())
    hard_pairs = {
        "T02_recall": _recall("T02"),
        "T02_as_T01_share": _share_as("T02", "T01"),
        "T05_recall": _recall("T05"),
        "T05_as_T04_share": _share_as("T05", "T04"),
        "T11_recall": _recall("T11"),
        "T11_as_T01_share": _share_as("T11", "T01"),
        "T11_as_T02_share": _share_as("T11", "T02"),
    }

    # -- write files
    (output / "metrics.json").write_text(json.dumps({
        "dummy": dummy_metrics,
        "model": {"val": val_metrics, "test": test_metrics},
        "joint_exact": joint_exact,
        "hard_pairs": hard_pairs,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "n_holdout_2026": len(holdout),
    }, indent=2))
    (output / "classification_report.json").write_text(
        json.dumps(report, indent=2))
    pd.DataFrame(cmat, index=labels, columns=labels).to_csv(
        output / "confusion_matrix.csv", index_label="actual\\predicted")
    routed["curves"].to_csv(output / "coverage_curve.csv", index=False)
    (output / "north_star.json").write_text(json.dumps(ns, indent=2))
    if routed["teams"] is not None:
        routed["teams"].to_csv(output / "per_team_routing.csv", index=False)
    (output / "issue_metrics.json").write_text(
        json.dumps(issue_metrics, indent=2))
    (output / "classification_report_issue.json").write_text(
        json.dumps(report_issue, indent=2))

    # per-team thresholds table + routing summary
    ptt_df = routed["routing_ptt"]["per_team"]
    ptt_df.to_csv(output / "per_team_thresholds.csv", index=False)
    (output / "per_team_thresholds_summary.json").write_text(json.dumps({
        "coverage": routed["routing_ptt"]["coverage"],
        "precision": routed["routing_ptt"]["precision"],
        "coverage_95ci": routed["routing_ptt"]["coverage_95ci"],
        "precision_95ci": routed["routing_ptt"]["precision_95ci"],
        "thresholds": routed["per_team_thresholds"],
    }, indent=2))

    # predictions_test.csv — baseline 2 columns + issue + per-team-threshold routing
    main_op = ns["operating_points"].get("at_95p") or {}
    global_thr = main_op.get("threshold")
    auto_routed = (test_conf >= global_thr) if global_thr is not None \
                  else np.zeros(len(test), dtype=bool)
    per_team_thr = routed["per_team_thresholds"]
    auto_routed_ptt = np.zeros(len(test), dtype=bool)
    for team, t in per_team_thr.items():
        if t is None:
            continue
        auto_routed_ptt |= (test_pred == team) & (test_conf >= t)

    preds_out = pd.DataFrame({
        C.ID_COL: test[C.ID_COL].astype(str).values,
        "actual_team_id": y_test,
        "actual_team_name": test["team_name"].values,
        "predicted_team_id": test_pred,
        "confidence": np.round(test_conf, 4),
        "auto_routed": auto_routed,
        "auto_routed_per_team": auto_routed_ptt,
        "issue_top1": i_top1,
        "issue_top2": i_top2,
        "issue_top3": i_top3,
        "actual_issue_id": y_test_i,
    })
    preds_out.to_csv(output / "predictions_test.csv", index=False)

    (output / "tuning.json").write_text(json.dumps(tuning, indent=2))

    # Save model with compaction
    model.compact()
    full_artifact = {
        "vectorizer": vec,
        "model": model,
        "team_classes": list(team_classes),
        "global_threshold": global_thr,
        "per_team_thresholds": per_team_thr,
        "issue_labels": issue_labels,
        "model_name": "improved_model_1",
    }
    model_path = output / "model.joblib"
    joblib.dump(full_artifact, model_path, compress=3)
    size_mb = model_path.stat().st_size / 1024**2

    run_info = {
        "model_name": "improved_model_1",
        "train_set": "train_cap_issue",
        "val_protocol": "shared",
        "best_C_team": tuning["best_team"]["C_team"],
        "best_C_issue": tuning["best_issue"]["C_issue"],
        "n_train": len(train), "n_val": len(val),
        "n_test": len(test), "n_holdout_2026": len(holdout),
        "wall_clock_sec": round(wall_clock_sec, 1),
        "artifact_size_mb": round(size_mb, 2),
        "global_threshold": global_thr,
        "per_team_thresholds": per_team_thr,
        "gpu_info": gpu_info(),
        "sklearn_version": _sklearn_version(),
        "args": vars(args),
    }
    (output / "run_info.json").write_text(json.dumps(run_info, indent=2))

    # Log result row
    log_result("improved_model_1", "test", {
        "train_set": "train_cap_issue",
        "team_acc": test_metrics["accuracy"],
        "team_macro_f1": test_metrics["macro_f1"],
        "issue_macro_f1": issue_metrics["macro_f1_predicted_team"],
        "joint_exact": joint_exact,
        "north_star_coverage_95": (ns["operating_points"].get("at_95p") or {}).get("coverage"),
        "precision_at_95_threshold": (ns["operating_points"].get("at_95p") or {}).get("precision"),
        "per_team_thr_coverage_95": routed["routing_ptt"]["coverage"],
        "teams_below_floor": ",".join((ns.get("guardrail") or {}).get("teams_below_floor", []) or []),
        "ece": ns.get("ece_test"),
        "n_eval": len(test),
    })

    # Enforce size gate
    if size_mb > MAX_ARTIFACT_MB:
        raise RuntimeError(
            f"model.joblib is {size_mb:.1f} MB > {MAX_ARTIFACT_MB} MB budget")

    if verbose:
        _print_summary(dummy_metrics, val_metrics, test_metrics, ns,
                       issue_metrics, routed["routing_ptt"], size_mb)


def _sklearn_version() -> str:
    import sklearn
    return sklearn.__version__


def _print_summary(dummy, val, test, ns, issue, ptt, size_mb) -> None:
    print("\n[results]")
    print(f"  {'metric':<14}{'dummy(val)':>12}{'model(val)':>12}"
          f"{'dummy(test)':>13}{'model(test)':>13}")
    for k in val:
        print(f"  {k:<14}{dummy['val'][k]:>12.3f}{val[k]:>12.3f}"
              f"{dummy['test'][k]:>13.3f}{test[k]:>13.3f}")
    main_op = ns["operating_points"].get("at_95p") or {}
    if main_op:
        lo, hi = main_op.get("coverage_95ci", [0, 0])
        plo, phi = main_op.get("precision_95ci", [0, 0])
        print(f"\n  [north star] coverage {main_op.get('coverage'):.1%} "
              f"(95% CI {lo:.1%}-{hi:.1%}) at precision "
              f"{main_op.get('precision'):.1%} (95% CI {plo:.1%}-{phi:.1%}), "
              f"threshold {main_op.get('threshold')}")
    print(f"  [per-team thr] coverage {ptt['coverage']:.1%} "
          f"precision {ptt['precision']:.1%}")
    print(f"  [issue] macro-F1 (pred team) = {issue['macro_f1_predicted_team']:.3f} "
          f"| oracle = {issue['macro_f1_oracle_team']:.3f}")
    print(f"  [artifact] model.joblib = {size_mb:.1f} MB")
    guardrail = ns.get("guardrail") or {}
    below = guardrail.get("teams_below_floor", []) or []
    print(f"  [guardrail] teams below 0.85 floor: {below if below else 'none'}")


# ------------------------ reference models -----------------------------------

def _evaluate_reference_models(train: pd.DataFrame, val: pd.DataFrame,
                               test: pd.DataFrame, holdout: pd.DataFrame,
                               verbose: bool) -> None:
    if verbose:
        print("\n[refs] flat_issue_ablation (word-only TF-IDF, LinearSVC "
              "on issue; team via mapping)", flush=True)
    flat_vec = build_vectorizer(use_char=False, word_max=100_000)
    X_tr = flat_vec.fit_transform(train[C.TEXT_COL].astype(str))
    X_te = flat_vec.transform(test[C.TEXT_COL].astype(str))
    flat = LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=5000)
    flat.fit(X_tr, train[C.ISSUE_COL])
    pred_issue = flat.predict(X_te)
    pred_team = np.array([issue_to_team.get(i, "T01") for i in pred_issue])
    log_result("flat_issue_ablation", "test", {
        "train_set": "train_cap_issue",
        "team_macro_f1": float(f1_score(test[C.TEAM_COL], pred_team,
                                        average="macro", zero_division=0)),
        "issue_macro_f1": float(f1_score(test[C.ISSUE_COL], pred_issue,
                                          average="macro", zero_division=0)),
        "n_eval": len(test),
    })

    if verbose:
        print("[refs] nb_reference_cap_issue (baseline 2 recipe on cap_issue)",
              flush=True)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                   min_df=2, max_df=0.98, max_features=75_000,
                                   sublinear_tf=True, strip_accents="unicode")),
        ("clf", MultinomialNB(alpha=0.1, fit_prior=False)),
    ])
    sw = compute_sample_weight(class_weight="balanced", y=train[C.TEAM_COL])
    pipe.fit(train[C.TEXT_COL].astype(str), train[C.TEAM_COL],
             clf__sample_weight=sw)
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
    cal.fit(val[C.TEXT_COL].astype(str), val[C.TEAM_COL])
    pred = cal.predict(test[C.TEXT_COL].astype(str))
    log_result("nb_reference_cap_issue", "test", {
        "train_set": "train_cap_issue",
        "team_macro_f1": float(f1_score(test[C.TEAM_COL], pred,
                                        average="macro", zero_division=0)),
        "n_eval": len(test),
    })

    if verbose:
        print("[refs] dummy_most_frequent", flush=True)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train[[C.TEXT_COL]], train[C.TEAM_COL])
    dpred = dummy.predict(test[[C.TEXT_COL]])
    log_result("dummy_most_frequent", "test", {
        "train_set": "train_cap_issue",
        "team_macro_f1": float(f1_score(test[C.TEAM_COL], dpred,
                                        average="macro", zero_division=0)),
        "n_eval": len(test),
    })

    if verbose:
        print("[refs] baseline_model_2 on 2026 holdout (committed model)",
              flush=True)
    b2 = joblib.load(C.BASELINE2_METRICS / "model.joblib")
    hpred = b2.predict(holdout[C.TEXT_COL].astype(str))
    log_result("baseline_model_2", "holdout_2026", {
        "train_set": "3k_team",
        "team_macro_f1": float(f1_score(holdout[C.TEAM_COL], hpred,
                                        average="macro", zero_division=0)),
        "n_eval": len(holdout),
    })


# -------------------------------- main ---------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-char", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--skip-refs", action="store_true",
                   help="Skip the reference model runs.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    verbose = not args.quiet
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if verbose:
        print(f"[load] splits from {C.SPLITS_V2}", flush=True)
    splits = _load_all_splits()
    train, val, test, holdout = (splits["train"], splits["val"],
                                 splits["test"], splits["holdout_2026"])
    names = dict(zip(pd.concat([train, val, test])[C.TEAM_COL],
                     pd.concat([train, val, test])["team_name"]))
    if verbose:
        print(f"[data] train={len(train):,}  val={len(val):,}  "
              f"test={len(test):,}  holdout={len(holdout):,}", flush=True)

    use_char = (not args.no_char) and USE_CHAR_DEFAULT
    vec = build_vectorizer(use_char=use_char,
                           word_max=WORD_MAX_FEATURES,
                           char_max=CHAR_MAX_FEATURES)
    if verbose:
        print(f"[vec] use_char={use_char} word_max={WORD_MAX_FEATURES} "
              f"char_max={CHAR_MAX_FEATURES}", flush=True)

    if verbose:
        print("[tune] grid search on validation", flush=True)
    C_t, C_i, tuning = _tune(vec, train, val, verbose=verbose)
    if verbose:
        print(f"[tune] best C_team={C_t} C_issue={C_i}", flush=True)

    if verbose:
        print("[fit] refitting on train with best Cs", flush=True)
    X_train = vec.fit_transform(train[C.TEXT_COL].astype(str))
    model = HierarchicalSVM(C_team=C_t, C_issue=C_i).fit(
        X_train, train[C.TEAM_COL].to_numpy(),
        train[C.ISSUE_COL].to_numpy())

    routed = _calibrate_and_route(model, vec, val, test, names, verbose)
    _write_artifacts(model, vec, train, val, test, holdout, routed,
                     tuning, args, time.time() - t0, verbose)

    if not args.skip_refs:
        _evaluate_reference_models(train, val, test, holdout, verbose)

    if verbose:
        print(f"\n[done] total {time.time() - t0:.1f}s "
              f"-> {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
