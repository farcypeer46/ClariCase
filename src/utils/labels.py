"""Team and issue label mappings, plus canonicalization for merged issues."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import (
    ISSUE_MAPPING, ISSUE_VARIANT_MERGES, MERGE_ISSUE_VARIANTS,
    METRICS_ROOT, PROCESSED, TEAM_MAPPING,
)


def load_issue_mapping() -> pd.DataFrame:
    return pd.read_csv(ISSUE_MAPPING)


def canonical_issue(product_issue_id: str) -> str:
    if not MERGE_ISSUE_VARIANTS:
        return product_issue_id
    return ISSUE_VARIANT_MERGES.get(product_issue_id, product_issue_id)


def _build_tables() -> tuple[pd.DataFrame, dict, dict, dict]:
    df = load_issue_mapping().copy()
    for col in ("product_issue_id", "team_id"):
        if col not in df.columns:
            raise KeyError(
                f"issue_mapping.csv missing '{col}'; got {df.columns.tolist()}")
    df["issue_id"] = df["product_issue_id"].map(canonical_issue)

    if "product_issue_label" in df.columns and "issue_label" not in df.columns:
        df = df.rename(columns={"product_issue_label": "issue_label"})

    canonical_cols = ["product_issue_id", "issue_id", "team_id"]
    if "issue_label" in df.columns:
        canonical_cols.append("issue_label")
    out = df[canonical_cols].drop_duplicates()

    issue_to_team = dict(zip(out["issue_id"], out["team_id"]))
    team_to_issues: dict[str, list[str]] = {}
    for tid, sub in out.groupby("team_id"):
        team_to_issues[tid] = sorted(sub["issue_id"].unique())

    team_df = pd.read_csv(TEAM_MAPPING)
    team_names = dict(zip(team_df["team_id"], team_df["team_name"]))
    return out, issue_to_team, team_to_issues, team_names


_canonical_df, issue_to_team, team_to_issues, team_names = _build_tables()

issue_labels: dict[str, str] = (
    dict(zip(_canonical_df["issue_id"], _canonical_df["issue_label"]))
    if "issue_label" in _canonical_df.columns else {}
)


def team_issue_mask(issue_classes: list[str],
                    team_classes: list[str]) -> np.ndarray:
    mask = np.zeros((len(team_classes), len(issue_classes)), dtype=bool)
    t_index = {t: i for i, t in enumerate(team_classes)}
    for j, iid in enumerate(issue_classes):
        t = issue_to_team.get(iid)
        if t in t_index:
            mask[t_index[t], j] = True
    return mask


def write_canonical_csv() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    _canonical_df.to_csv(PROCESSED / "issue_canonical.csv", index=False)
    tracked = METRICS_ROOT / "improved_model_1"
    tracked.mkdir(parents=True, exist_ok=True)
    _canonical_df.to_csv(tracked / "issue_canonical.csv", index=False)


if __name__ == "__main__":
    write_canonical_csv()
    print(f"[labels] {len(_canonical_df)} issues; "
          f"{len(team_to_issues)} teams; canonical CSV written")
