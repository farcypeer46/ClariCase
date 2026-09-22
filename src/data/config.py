from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"

RAW_FILE = "complaints.csv"

TEXT = "Consumer complaint narrative"
DATE = "Date received"
LABEL = "Product"

# Excluded, with reasons checked against this export:
#   Company public response     -- 100% null
#   Consumer disputed?          -- 100% null, discontinued field
#   Consumer consent provided?  -- constant ("Consent provided"), no information

KEEP = [DATE, LABEL, "Sub-product", "Issue", "Sub-issue", TEXT,
        "Company", "State", "ZIP code", "Tags", "Submitted via",
        "Date sent to company", "Company response to consumer",
        "Timely response?", "Complaint ID"]