"""EDA. Writes docs/data_profile.md.

Run: python -m src.data.explore

"""

import re

import pandas as pd

from src.data.config import DOCS, TEXT, DATE, LABEL
from src.data.ingest import load_raw

XXXX = re.compile(r"X{2,}")


def canonical(s: pd.Series) -> pd.Series:
    """Aggressive normalisation, used only to count duplicates."""
    return (s.astype(str).str.lower()
             .str.replace(XXXX, " ", regex=True)
             .str.replace(r"[^a-z0-9 ]", " ", regex=True)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip())

OUT = DOCS / "data_profile.md"


def build(df: pd.DataFrame) -> str:
    md, w = [], None
    w = md.append

    w("# Data profile\n")
    w(f"- Rows: **{len(df):,}**  |  Columns: {len(df.columns)}")
    w(f"- Dates: {df[DATE].min().date()} to {df[DATE].max().date()}")
    w(f"- Companies: {', '.join(df['Company'].unique()[:5])}")
    w(f"- Channels: {dict(df['Submitted via'].value_counts())}\n")

    w("## Rows per month\n")
    w("A thin final month is normal -- complaints publish only after the company")
    w("responds or 15 days pass, so the tail of any snapshot is partial.\n")
    w(df["month"].value_counts().sort_index().to_frame("n").to_markdown() + "\n")

    w("## Product distribution\n")
    vc = df[LABEL].value_counts()
    w(pd.DataFrame({"n": vc, "pct": (vc / len(df) * 100).round(1)}).to_markdown() + "\n")

    w("## Product x month\n")
    w("Look for a class that stops or starts. That is either a taxonomy change")
    w("or a real shift in what gets filed -- both matter for a temporal split.\n")
    w(pd.crosstab(df["month"], df[LABEL]).to_markdown() + "\n")

    w("## Finer labels\n")
    for c in ["Sub-product", "Issue", "Sub-issue"]:
        if c in df.columns:
            w(f"- `{c}`: {df[c].nunique()} distinct")
    w("")

    w("## Narrative length\n")
    tok = df[TEXT].str.split().str.len()
    w(pd.DataFrame({"chars": df[TEXT].str.len(), "tokens": tok})
        .describe(percentiles=[.5, .9, .95, .99]).round(0).to_markdown() + "\n")
    for limit in (128, 256, 512):
        w(f"- Over {limit} tokens: **{(tok > limit).mean() * 100:.1f}%**")
    w("\nThat last figure sets the truncation length for a transformer. "
      "Worth ablating 256 against 512 rather than assuming.\n")

    w("## Redactions\n")
    has = df[TEXT].str.contains(r"X{2,}", regex=True)
    cnt = df[TEXT].str.count(r"X{2,}")
    w(f"- Contain a redaction: **{has.mean() * 100:.1f}%**")
    w(f"- Median per narrative: {cnt.median():.0f}  |  max: {cnt.max():.0f}\n")

    w("## Null rate\n")
    nulls = (df.isna().mean() * 100).round(1)
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    w(nulls.to_frame("pct_null").to_markdown() if len(nulls) else "No nulls.")
    w("")
    if len(high := nulls[nulls > 20]):
        w(f"**Over 20% null:** {', '.join(high.index)} — check why before "
          "building anything on these.\n")

    w("## Duplicates\n")
    c = canonical(df[TEXT])
    dup = c.duplicated(keep=False)
    w(f"- Rows in a repeated group: **{dup.sum():,}** ({dup.mean() * 100:.1f}%)")
    w(f"- Distinct texts: {c.nunique():,}\n")
    w("Largest groups:\n")
    for i, (t, n) in enumerate(c.value_counts().head(5).items(), 1):
        w(f"{i}. **{n} copies** — `{t[:160]}...`")
    w("")

    w("## Leakage check\n")
    w("Columns that describe what happened *after* routing must never become")
    w("features. Listed here so the exclusion is deliberate, not accidental.\n")
    for c_ in ["Company response to consumer", "Timely response?",
               "Company public response", "Consumer disputed?"]:
        if c_ in df.columns:
            w(f"- `{c_}` — post-hoc outcome, exclude")
    w("")



    return "\n".join(md)


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(load_raw()))
    print(f"[write] {OUT}")