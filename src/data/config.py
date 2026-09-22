"""Paths and column names. 
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"

RAW_FILE = "complaints.csv"

TEXT = "Consumer complaint narrative"
DATE = "Date received"
LABEL = "Product"


DATE_MIN = "2024-01-01"
DATE_MAX = "2026-02-01"


MIN_TOKENS = 5


DROP_CLASSES = {
    "Student loan",                                            # 4
    "Prepaid card",                                            # 25
    "Payday loan, title loan, personal loan, or advance loan",  # 35
    "Debt or credit management",                               # 36
}


# Train before this date, test on and after. 
SPLIT_AT = "2025-07-01"


THIN_CLASS_WARN = 200

# Post-hoc outcomes. Never features  they describe what happened after
# routing. Tags is kept for the fairness audit but is also never a feature.
NEVER_FEATURES = ["Company response to consumer", "Timely response?", "Tags"]

# Excluded, with reasons checked against this export:
#   Company public response     -- 100% null
#   Consumer disputed?          -- 100% null, discontinued field
#   Consumer consent provided?  -- constant ("Consent provided"), no information
#
# Tags is 84% null but carries "Older American" and "Servicemember" on ~3,500
# rows. It is the only protected-group signal in the file, so keep it for the
# fairness audit. It is NOT a feature never train on it.
KEEP = [DATE, LABEL, "Sub-product", "Issue", "Sub-issue", TEXT,
        "Company", "State", "ZIP code", "Tags", "Submitted via",
        "Date sent to company", "Company response to consumer",
        "Timely response?", "Complaint ID"]