"""Pick short example complaints for the structured team descriptions.

Examples come only from the training split, PII-masked like every other Jev
input, and are saved to structured_examples.json so the question sent to Jev
is identical on every run and can be reviewed before any request is made.

    python -m src.jev_model.examples                    # structured: 3 x 7 teams
    python -m src.jev_model.examples --variant rich     # rich: 5 x weak teams
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.app.pii import mask
from src.jev_model.config import (RICH_EXAMPLES_FILE, SEED, STRUCTURED_TEAMS,
                                  WEAK_TEAMS)

EXAMPLES_FILE = Path(__file__).with_name("structured_examples.json")
SOURCE_SPLIT = "train_cap_issue"
PER_TEAM = 3
# Short enough to keep requests cheap, long enough to show the product.
MIN_WORDS, MAX_WORDS = 25, 70

# A good example carries the team's label AND names the team's core product,
# so it illustrates the team rather than a borderline case.
CORE_TERMS: dict[str, str] = {
    "T01": r"credit report|equifax|experian|transunion|credit bureau",
    "T02": r"debt collector|collection agency|collections agency|collector",
    "T04": r"checking account|savings account|overdraft",
    "T05": r"zelle|venmo|cash app|paypal|wire transfer|western union|crypto|bitcoin",
    "T09": r"payday|title loan|personal loan|installment loan|cash advance",
    "T10": r"prepaid|gift card|benefit card|payroll card",
    "T11": r"credit repair|debt settlement|debt relief|credit counsel",
    "T03": r"credit card",
    "T06": r"mortgage|escrow|foreclos|loan modification",
    "T07": r"student loan",
    "T08": r"car loan|auto loan|vehicle|lease|repossess",
}


def pick_examples(train: pd.DataFrame, per_team: int = PER_TEAM,
                  seed: int = SEED, teams=STRUCTURED_TEAMS) -> dict[str, list[dict]]:
    out = {}
    for team in teams:
        rows = train[train["team_id"] == team].copy()
        rows["text"] = rows["complaint_text"].astype(str).map(mask)
        words = rows["text"].str.split().str.len()
        on_topic = rows["text"].str.contains(CORE_TERMS[team], case=False, regex=True)
        rows = rows[(words >= MIN_WORDS) & (words <= MAX_WORDS) & on_topic]
        if len(rows) < per_team:
            raise ValueError(f"Only {len(rows)} short training examples for {team}")
        chosen = rows.sample(n=per_team, random_state=seed)
        out[team] = [{"complaint_id": str(r.complaint_id), "text": r.text}
                     for r in chosen.itertuples()]
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=("structured", "rich"), default="structured")
    variant = ap.parse_args(argv).variant
    if variant == "rich":
        teams, per_team, path = WEAK_TEAMS, 5, RICH_EXAMPLES_FILE
    else:
        teams, per_team, path = STRUCTURED_TEAMS, PER_TEAM, EXAMPLES_FILE

    from src.utils.data import load_split
    examples = pick_examples(load_split(SOURCE_SPLIT), per_team, SEED, teams)
    path.write_text(json.dumps(
        {"source_split": SOURCE_SPLIT, "seed": SEED, "per_team": per_team,
         "examples": examples}, indent=2) + "\n")
    print(f"[write] {path} "
          f"({sum(len(v) for v in examples.values())} examples, "
          f"{len(examples)} teams)")


if __name__ == "__main__":
    main()
