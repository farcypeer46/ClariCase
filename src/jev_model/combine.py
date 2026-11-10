"""Free analyses on cached Jev answers (no API calls).

1. Second opinion: can Jev raise Improved Model 1's auto-routing rate at the
   95% precision target? Rule: route a complaint when Improved Model 1's
   confidence clears one threshold, or a lower threshold when Jev picks the
   same team. Both thresholds are fit on the validation sample and frozen
   for test, like every other operating point in the project.
2. Calibration: how far isotonic recalibration (fit on validation) reduces
   Jev's calibration error. It cannot change which complaints rank highest,
   so it does not change the best precision any threshold can reach.
3. Per-team thresholds: a separate 95%-precision threshold for each team,
   fit on validation, so Jev can auto-route the teams it is precise on even
   when one overall threshold cannot reach 95%. Improved Model 1 is scored the
   same way for comparison. A team needs MIN_ROUTED complaints routed to it on
   validation to get a threshold; smaller teams are left to manual review.

    python -m src.jev_model.combine                     # zero-shot answers
    python -m src.jev_model.combine --variant structured
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.jev_model.config import (ARTIFACT_DIR, MODEL, N_TEST_DEFAULT,
                                  N_VAL_DEFAULT, SEED, TEAM_IDS, VARIANTS,
                                  cache_path, question_sha)
from src.jev_model.evaluate import (classify_frame, improved_model_1_proba,
                                    sample_split)
from src.utils.config import MIN_ROUTED
from src.utils.labels import team_names
from src.utils.routing import (apply_per_team_thresholds,
                                expected_calibration_error,
                                per_team_thresholds)

TARGET = 0.95
# Same 0.002 step as the project north-star harness (501 thresholds).
GRID = np.round(np.linspace(0.0, 1.0, 501), 3)


class CacheOnly:
    """Fails loudly if any answer is missing from the cache: never pays."""
    model = MODEL

    def classify_team(self, state):
        raise RuntimeError("Answer not cached; run evaluate.py first.")


def route_mask(im1_conf, agree, t_agree, t_alone):
    return np.where(agree, im1_conf >= t_agree, im1_conf >= t_alone)


def fit_rule(im1_conf, agree, correct):
    """Thresholds with the most coverage at >= TARGET precision on validation."""
    best = None
    for t_alone in GRID:
        for t_agree in GRID[GRID <= t_alone]:
            r = route_mask(im1_conf, agree, t_agree, t_alone)
            if r.sum() < MIN_ROUTED or correct[r].mean() < TARGET:
                continue
            if best is None or r.mean() > best[0]:
                best = (r.mean(), float(t_agree), float(t_alone))
    return best


def stats(mask, correct):
    return {"coverage": round(float(mask.mean()), 4),
            "precision": round(float(correct[mask].mean()), 4) if mask.any() else None,
            "n_routed": int(mask.sum())}


def per_team(v, t, model: str) -> dict:
    thr = per_team_thresholds(v[f"{model}_conf"], v[f"{model}_pred"], v["truth"],
                              target=TARGET)
    res = apply_per_team_thresholds(t["truth"], t[f"{model}_pred"],
                                    t[f"{model}_conf"], thr, team_names)
    res["per_team"] = res["per_team"].to_dict(orient="records")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=VARIANTS, default="zeroshot")
    variant = ap.parse_args(argv).variant

    from src.utils.data import load_split
    val = sample_split(load_split("val"), N_VAL_DEFAULT)
    test = sample_split(load_split("test"), N_TEST_DEFAULT, seed=SEED + 1)
    cache = cache_path(MODEL, variant)
    teams = np.array(TEAM_IDS)

    split = {}
    for name, df in (("val", val), ("test", test)):
        jev = classify_frame(CacheOnly(), df, cache, qsha=question_sha(variant),
                             legacy_ok=variant == "zeroshot")
        im1 = improved_model_1_proba(df["model_text"].tolist())
        truth = df["team_id"].to_numpy()
        split[name] = {
            "truth": truth,
            "im1_conf": im1.max(1), "im1_pred": teams[im1.argmax(1)],
            "jev_conf": jev.max(1), "jev_pred": teams[jev.argmax(1)],
            "agree": im1.argmax(1) == jev.argmax(1)}
        split[name]["im1_ok"] = split[name]["im1_pred"] == truth
        split[name]["jev_ok"] = split[name]["jev_pred"] == truth
    v, t = split["val"], split["test"]

    # Improved Model 1 alone: the same rule with no agreement discount.
    alone = fit_rule(v["im1_conf"], np.zeros_like(v["agree"]), v["im1_ok"])
    combo = fit_rule(v["im1_conf"], v["agree"], v["im1_ok"])
    out = {"variant": variant, "target_precision": TARGET,
           "n_val": len(val), "n_test": len(test)}
    for label, rule in (("improved_model_1_alone", alone),
                        ("second_opinion", combo)):
        _, t_agree, t_alone = rule
        if label == "second_opinion":
            agree = t["agree"]
        else:   # one threshold for every complaint; Jev is not consulted
            agree, t_agree = np.zeros_like(t["agree"]), t_alone
        out[label] = {"threshold_when_jev_agrees": t_agree,
                      "threshold_otherwise": t_alone,
                      "test": stats(route_mask(t["im1_conf"], agree, t_agree,
                                               t_alone), t["im1_ok"])}

    iso = IsotonicRegression(out_of_bounds="clip").fit(v["jev_conf"], v["jev_ok"])
    out["jev_calibration"] = {
        "ece_test_raw": round(expected_calibration_error(t["jev_conf"], t["jev_ok"]), 4),
        "ece_test_isotonic": round(expected_calibration_error(
            iso.predict(t["jev_conf"]), t["jev_ok"]), 4)}

    out["per_team_thresholds"] = {"jev": per_team(v, t, "jev"),
                                  "improved_model_1": per_team(v, t, "im1")}

    out_dir = ARTIFACT_DIR if variant == "zeroshot" else ARTIFACT_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "second_opinion.json").write_text(
        json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item")
                   else None))
    print(f"[write] {out_dir / 'second_opinion.json'}")


if __name__ == "__main__":
    main()
