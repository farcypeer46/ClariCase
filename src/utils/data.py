"""Readers for persisted splits and the 2026 holdout cohort."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from src.utils.config import HOLDOUT_PARQUET, HOLDOUT_ZIP, SPLITS_V2
from src.utils.labels import canonical_issue

_VALID_SPLITS = {"train_full", "train_cap_issue", "train_3k_team", "val", "test"}


def load_split(name: str) -> pd.DataFrame:
    if name not in _VALID_SPLITS:
        raise ValueError(f"Unknown split {name!r}; want one of {_VALID_SPLITS}")
    return pd.read_parquet(SPLITS_V2 / f"{name}.parquet")


def _extract_holdout() -> Path:
    HOLDOUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    if HOLDOUT_PARQUET.exists():
        return HOLDOUT_PARQUET
    with zipfile.ZipFile(HOLDOUT_ZIP) as zf:
        member = next(n for n in zf.namelist()
                      if n.endswith("fresh_holdout_2026_01_02.parquet"))
        zf.extract(member, HOLDOUT_PARQUET.parent)
        extracted = HOLDOUT_PARQUET.parent / member
        if extracted != HOLDOUT_PARQUET:
            extracted.rename(HOLDOUT_PARQUET)
    return HOLDOUT_PARQUET


def load_holdout_2026() -> pd.DataFrame:
    path = _extract_holdout()
    df = pd.read_parquet(path)
    required = {"complaint_id", "complaint_text", "team_id", "product_issue_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"holdout missing {missing}; got {df.columns.tolist()}")
    df["issue_id"] = df["product_issue_id"].map(canonical_issue)
    return df
