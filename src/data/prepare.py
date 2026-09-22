"""Clean, filter, dedup, split.

Thresholds live in src/data/config.py. Writes nothing to disk.
"""

import re
import unicodedata

import pandas as pd

from src.data.config import (TEXT, DATE, LABEL, DATE_MIN, DATE_MAX,
                             MIN_TOKENS, DROP_CLASSES, SPLIT_AT,
                             THIN_CLASS_WARN, NEVER_FEATURES)
from src.data.ingest import load_raw

XXXX = re.compile(r"X{2,}")


def clean_text(s: pd.Series) -> pd.Series:
   
    return (s.astype(str)
             .map(lambda x: unicodedata.normalize("NFKC", x))
             .str.replace(r"\s+", " ", regex=True)
             .str.strip())


def canonical(s: pd.Series) -> pd.Series:
    """Aggressive normalisation, used only to detect duplicates."""
    return (s.str.lower()
             .str.replace(XXXX, " ", regex=True)
             .str.replace(r"[^a-z0-9 ]", " ", regex=True)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip())


def prepare(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    n = len(df)

    def step(name: str) -> None:
        nonlocal n
        if verbose:
            print(f"  {name:<24} {n:>7,} -> {len(df):>7,}  (-{n - len(df):,})")
        n = len(df)

    if verbose:
        print(f"\n[prepare] starting from {n:,} rows")

    df = df[df[TEXT].notna() & (df[TEXT].astype(str).str.strip() != "")]
    step("has narrative")

    df = df[(df[DATE] >= DATE_MIN) & (df[DATE] < DATE_MAX)].copy()
    step(f"window {DATE_MIN[:7]}..{DATE_MAX[:7]}")

    df[TEXT] = clean_text(df[TEXT])
    df["n_tokens"] = df[TEXT].str.split().str.len()
    df = df[df["n_tokens"] >= MIN_TOKENS]
    step(f"min {MIN_TOKENS} tokens")

    dropped = df[df[LABEL].isin(DROP_CLASSES)][LABEL].value_counts()
    df = df[~df[LABEL].isin(DROP_CLASSES)]
    step("rare classes")
    if verbose and len(dropped[dropped > 0]):
        print(f"       dropped: {dict(dropped[dropped > 0])}")

    
    df["_canon"] = canonical(df[TEXT])
    df["dup_group_size"] = df.groupby("_canon")[LABEL].transform("size")
    df = df.drop_duplicates("_canon", keep="first").drop(columns="_canon")
    step("exact duplicates")

    df[LABEL] = df[LABEL].astype(str).astype("category")
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df[DATE] < SPLIT_AT].copy()
    test = df[df[DATE] >= SPLIT_AT].copy()

    # Dedup ran before the split, so nothing can straddle the boundary.
    assert set(train["Complaint ID"]).isdisjoint(test["Complaint ID"]), "ID overlap"
    assert not set(canonical(train[TEXT])) & set(canonical(test[TEXT])), "text overlap"

    if not verbose:
        return train, test

    print(f"\n[split] at {SPLIT_AT} (temporal, not random)")
    print(f"  train {len(train):>6,}  {train[DATE].min().date()} to {train[DATE].max().date()}")
    print(f"  test  {len(test):>6,}  {test[DATE].min().date()} to {test[DATE].max().date()}"
          f"   [{len(test) / len(df):.0%}]\n")

    counts = pd.DataFrame({"train": train[LABEL].value_counts(),
                           "test": test[LABEL].value_counts()})
    print(counts.to_string())

    thin = counts[counts["test"] < THIN_CLASS_WARN]
    if len(thin):
        print(f"\n[warn] under {THIN_CLASS_WARN} test examples: "
              f"{', '.join(thin.index)}")
        print("       per-class recall on these swings between runs — "
              "report it as noisy")

    print(f"[note] never use as features: {', '.join(NEVER_FEATURES)}")
    return train, test


def get_splits(verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    return split(prepare(load_raw(), verbose), verbose)


if __name__ == "__main__":
    get_splits(verbose=True)