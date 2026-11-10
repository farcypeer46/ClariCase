"""Settings and the team question sent to Jev.

Jev cannot be fine-tuned on customer data, so it is steered only by the
question. Two versions of the question exist:

- "zeroshot": each team is a one-line description of its product and issues.
- "structured": for the teams Jev confused in the zero-shot run, the
  description becomes an object with what the team covers, what belongs to a
  neighbouring team instead, and a few training-set example complaints, as
  TypeSafe's docs recommend for options the model keeps confusing
  (https://docs.typesafe.ai/primitives/choice.md). Other teams keep the
  one-liner, and the instructions are identical, so any difference in
  results comes from the descriptions alone.

API reference: https://docs.typesafe.ai/api/system-one.md
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

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
EXAMPLES_FILE = Path(__file__).with_name("structured_examples.json")
RICH_EXAMPLES_FILE = Path(__file__).with_name("rich_examples.json")

N_VAL_DEFAULT = 2_000
N_TEST_DEFAULT = 2_000
SEED = RANDOM_SEED
CONCURRENCY = 4
TIMEOUT_SEC = 30
MAX_RETRIES = 5

QUESTION_NAME = "team"
TEAM_IDS: tuple[str, ...] = tuple(sorted(team_to_issues))
VARIANTS = ("zeroshot", "structured", "rich")

INSTRUCTIONS = (
    "This is a consumer complaint about a financial product or service. "
    "Which support team should handle it? Choose the team whose product the "
    "complaint is about. Redacted details appear as XXXX.")

# Boundaries for the teams Jev confused in the zero-shot run: Debt Collection
# vs Credit Reporting (64 misroutes), Banking vs Money Transfers (44), and the
# small teams it over-predicted (Debt & Credit Management, Personal &
# Short-Term Loans, Prepaid Card).
STRUCTURED_TEAMS: dict[str, dict[str, str]] = {
    "T01": {
        "covers": "Errors or disputes about what appears on a credit report, "
                  "how a credit bureau investigated a dispute, identity theft "
                  "entries, and who accessed or used a credit report.",
        "not_for": "A collector contacting the consumer or demanding payment "
                   "goes to the Debt Collection Team, even when the debt also "
                   "shows on a credit report. Credit repair or debt "
                   "settlement companies go to the Debt & Credit Management "
                   "Team.",
    },
    "T02": {
        "covers": "A debt collector or collection agency contacting the "
                  "consumer, demanding payment, collecting a debt the "
                  "consumer says is not owed, failing to validate a debt, or "
                  "threatening action.",
        "not_for": "A dispute with a credit bureau about an entry on a "
                   "credit report goes to the Credit Reporting Team, even if "
                   "the entry is a collection account. Billing by the "
                   "original lender goes to that product's team.",
    },
    "T04": {
        "covers": "Checking or savings accounts: opening, closing, fees, "
                  "overdrafts, deposits, holds, account access, and "
                  "unauthorized debit card transactions on the account.",
        "not_for": "Sending or receiving money through Zelle, Venmo, Cash "
                   "App, PayPal, wire transfers, international transfers or "
                   "crypto goes to the Money Transfers & Digital Currency "
                   "Team. Prepaid or gift cards go to the Prepaid Card Team.",
    },
    "T05": {
        "covers": "Person-to-person payment apps, wire and international "
                  "transfers, remittances, money orders, mobile wallets and "
                  "virtual currency.",
        "not_for": "Problems with the checking or savings account itself "
                   "(fees, overdrafts, closures, account access) go to the "
                   "Banking Accounts Team.",
    },
    "T09": {
        "covers": "Payday loans, title loans, installment personal loans and "
                  "cash advance apps.",
        "not_for": "Car loans go to the Vehicle Finance Team, student loans "
                   "to the Student Loan Team, credit cards to the Credit Card "
                   "Team, and a collector chasing an old loan to the Debt "
                   "Collection Team.",
    },
    "T10": {
        "covers": "Prepaid debit cards, gift cards, payroll cards and "
                  "government benefit cards.",
        "not_for": "Debit cards linked to a checking account go to the "
                   "Banking Accounts Team; credit cards go to the Credit Card "
                   "Team.",
    },
    "T11": {
        "covers": "Credit repair companies, debt settlement or debt relief "
                  "companies, and credit counseling services.",
        "not_for": "Disputes directly with a credit bureau go to the Credit "
                   "Reporting Team; collectors demanding payment go to the "
                   "Debt Collection Team.",
    },
}


# Teams furthest behind Improved Model 1 in the structured run (F1 gap):
# Debt & Credit Management -0.23, Personal Loans -0.17, Student Loan -0.12,
# Prepaid Card (lowest F1 overall). The "rich" variant gives these five
# examples each; every other team is described exactly as in "structured".
# Debt Collection is left out: its errors come from ambiguous labels.
WEAK_TEAMS: tuple[str, ...] = ("T07", "T09", "T10", "T11")

# Boundaries for teams that "structured" left as one-liners. Kept separate
# so the "structured" question (and its cached answers) is unchanged.
EXTRA_TEAMS: dict[str, dict[str, str]] = {
    "T03": {
        "covers": "Credit cards: billing disputes, fees and interest, "
                  "rewards, credit limits, card closures and unauthorized "
                  "charges on a credit card.",
        "not_for": "A collector chasing an old card debt goes to the Debt "
                   "Collection Team; an error on a credit report goes to the "
                   "Credit Reporting Team; debit or prepaid cards go to the "
                   "Banking Accounts or Prepaid Card Team.",
    },
    "T06": {
        "covers": "Mortgages and home equity loans: applications, payments "
                  "and escrow, servicing transfers, loan modifications, "
                  "foreclosure and closing costs.",
        "not_for": "Reverse mortgages and home equity lines also belong "
                   "here; a collector on a foreclosed balance goes to the "
                   "Debt Collection Team.",
    },
    "T07": {
        "covers": "Federal and private student loans: servicing, repayment "
                  "plans, forgiveness, deferment, and how the servicer "
                  "handles payments.",
        "not_for": "Personal loans used for school costs go to the Personal "
                   "& Short-Term Loans Team; a collector on a defaulted "
                   "loan goes to the Debt Collection Team.",
    },
    "T08": {
        "covers": "Car loans and vehicle leases: payments, interest, "
                  "repossession, lease-end charges, and dealer-arranged "
                  "financing.",
        "not_for": "Title loans that use a car as collateral go to the "
                   "Personal & Short-Term Loans Team.",
    },
}

# Yes/no ("noul") checks asked in the same request as the team question.
# Each is answered independently, so they do not change the team answer;
# they give a second signal that is combined with it in code.
NOUL_QUESTIONS: dict[str, str] = {
    "collector": "Is the consumer complaining about a debt collector or "
                 "collection agency contacting them, demanding payment, or "
                 "trying to collect a debt?",
    "credit_report": "Is the consumer disputing information on their credit "
                     "report, or how a credit bureau handled their dispute?",
    "payment_app": "Is the complaint about sending or receiving money "
                   "through a payment app, wire transfer, money transfer "
                   "service or digital currency?",
    "bank_account": "Is the complaint about a checking or savings account "
                    "itself, such as fees, overdrafts, deposits, closure or "
                    "access to the account?",
    "multi_product": "Does the complaint involve more than one financial "
                     "product, such as both a loan and a credit report?",
}


def _team_description(team_id: str) -> str:
    # "Debt collection :: Attempts to collect debt not owed" -> product, issue
    labels = [issue_labels[i] for i in team_to_issues[team_id]]
    product = labels[0].split(" :: ", 1)[0]
    issues = "; ".join(label.split(" :: ", 1)[-1] for label in labels)
    return f"{team_names[team_id]}. Product: {product}. Typical issues: {issues}."


def _load_examples(path: Path = EXAMPLES_FILE) -> dict[str, list[str]]:
    data = json.loads(path.read_text())
    return {t: [e["text"] for e in rows] for t, rows in data["examples"].items()}


def team_criteria(variant: str = "zeroshot") -> dict[str, str | dict]:
    """Choice criteria keyed by team ID, so answers map straight to labels."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; want one of {VARIANTS}")
    criteria: dict[str, str | dict] = {t: _team_description(t) for t in TEAM_IDS}
    if variant == "structured":
        examples = _load_examples()
        for team, bounds in STRUCTURED_TEAMS.items():
            criteria[team] = {"description": criteria[team], **bounds,
                              "examples": examples[team]}
    elif variant == "rich":
        criteria = team_criteria("structured")
        examples = _load_examples(RICH_EXAMPLES_FILE)
        bounds_for = {**STRUCTURED_TEAMS, **EXTRA_TEAMS}
        for team in WEAK_TEAMS:
            criteria[team] = {"description": _team_description(team),
                              **bounds_for[team], "examples": examples[team]}
    return criteria


def team_question(variant: str = "zeroshot") -> dict:
    questions = {QUESTION_NAME: {"type": "choice", "instructions": INSTRUCTIONS,
                                 "criteria": team_criteria(variant)}}
    if variant == "rich":
        for name, text in NOUL_QUESTIONS.items():
            questions[name] = {"type": "noul", "instructions": text}
    return questions


def question_sha(variant: str = "zeroshot") -> str:
    """Fingerprint of the exact question, stored with every cached answer."""
    blob = json.dumps(team_question(variant), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def cache_path(model: str, variant: str = "zeroshot") -> Path:
    # The zero-shot file keeps its original name so earlier answers stay valid.
    name = model if variant == "zeroshot" else f"{model}__{variant}"
    return CACHE_DIR / f"{name}.jsonl"
