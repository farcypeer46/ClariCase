"""Settings and the team question sent to Jev.

Jev cannot be fine-tuned on customer data, so it is used zero-shot: each team
is described by its product and the issues it handles, and Jev picks one.
API reference: https://docs.typesafe.ai/api/system-one.md
"""
from __future__ import annotations

import os

from src.utils.config import CACHE, METRICS_ROOT, RANDOM_SEED
from src.utils.labels import issue_labels, team_names, team_to_issues

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
# Override with JEV_MODEL. A pinned version (e.g. "jev-1.13.0") keeps
# thresholds valid across Jev releases; the answering version is recorded.
MODEL = os.environ.get("JEV_MODEL", "jev-latest")

# Published price at launch: input tokens only, output is free. Used for the
# pre-run cost estimate; the actual bill comes from TypeSafe's console.
PRICE_PER_M_INPUT_TOKENS = 0.042

ARTIFACT_DIR = METRICS_ROOT / "jev_model"
CACHE_DIR = CACHE / "jev_model"          # gitignored with the rest of data/cache

N_VAL_DEFAULT = 2_000
N_TEST_DEFAULT = 2_000
SEED = RANDOM_SEED
CONCURRENCY = 4
TIMEOUT_SEC = 30
MAX_RETRIES = 5

QUESTION_NAME = "team"
TEAM_IDS: tuple[str, ...] = tuple(sorted(team_to_issues))

INSTRUCTIONS = (
    "This is a consumer complaint about a financial product or service. "
    "Which support team should handle it? Choose the team whose product the "
    "complaint is about. Redacted details appear as XXXX.")


def _team_description(team_id: str) -> str:
    # "Debt collection :: Attempts to collect debt not owed" -> product, issue
    labels = [issue_labels[i] for i in team_to_issues[team_id]]
    product = labels[0].split(" :: ", 1)[0]
    issues = "; ".join(label.split(" :: ", 1)[-1] for label in labels)
    return f"{team_names[team_id]}. Product: {product}. Typical issues: {issues}."


def team_criteria() -> dict[str, str]:
    """Choice criteria keyed by team ID, so answers map straight to labels."""
    return {t: _team_description(t) for t in TEAM_IDS}


def team_question() -> dict:
    return {QUESTION_NAME: {"type": "choice", "instructions": INSTRUCTIONS,
                            "criteria": team_criteria()}}
