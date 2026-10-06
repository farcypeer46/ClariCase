"""Build temporal + group-aware splits for improved models.

Reads the 1.9 GB CSV once into a Parquet cache, then produces:

- `data/cache/splits_v2/train_full.parquet`     (774,412 rows — all train-period)
- `data/cache/splits_v2/train_cap_issue.parquet`(main training set, ≤10k/issue)
- `data/cache/splits_v2/train_3k_team.parquet`  (3k/team — baseline 2 recipe)
- `data/cache/splits_v2/train_full_ids.parquet` (just IDs, for word-vector corpus)
- `data/cache/splits_v2/val.parquet`            (153,502 rows, real mix)
- `data/cache/splits_v2/test.parquet`           (109,834 rows, real mix)
- `data/cache/splits_v2/split_report.json`      (counts, assertions, parity)

Parity: test complaint_ids match the committed baseline 2 predictions file.

Usage:  python -m src.utils.build_splits
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.utils.config import (
    BASELINE2_METRICS, CHUNKSIZE, DATE_COL, FULL_CSV, FULL_PARQUET, GROUP_COL,
    ID_COL, ISSUE_COL, METRICS_ROOT, PROCESSED, RANDOM_SEED, SPLITS_V2,
    TEAM_COL, TEST_END, TEXT_COL, TRAIN_CAP_PER_ISSUE, TRAIN_END,
    BASELINE2_CAP_PER_TEAM, USECOLS, VAL_END,
)
from src.utils.labels import canonical_issue

_OUT_COLS = [ID_COL, DATE_COL, "company", TEXT_COL, TEAM_COL, "team_name",
             ISSUE_COL, GROUP_COL]


def csv_to_parquet(verbose: bool = True) -> None:
    """Stream the CSV to Parquet. Idempotent."""
    if FULL_PARQUET.exists():
        if verbose:
            print(f"[parquet] cached {FULL_PARQUET}")
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    FULL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    reader = pd.read_csv(
        FULL_CSV, usecols=USECOLS, chunksize=CHUNKSIZE,
        dtype={ID_COL: "string"}, keep_default_na=False,
    )
    writer = None
    rows = 0
    try:
        for i, chunk in enumerate(reader, start=1):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(FULL_PARQUET, table.schema,
                                          compression="zstd")
            writer.write_table(table)
            rows += len(chunk)
            if verbose and i % 5 == 0:
                print(f"[parquet]  chunk {i:>3} ({rows:,} rows written)")
    finally:
        if writer is not None:
            writer.close()
    if verbose:
        print(f"[parquet] wrote {rows:,} rows -> {FULL_PARQUET}")


def _assign_periods(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.assign(_split=np.select(
        [dates < TRAIN_END, dates < VAL_END, dates < TEST_END],
        ["train", "val", "test"], default=""))
    return df[df["_split"] != ""]


def _drop_crossing_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    nunique = df.groupby(GROUP_COL)["_split"].transform("nunique")
    crossing = nunique > 1
    n_rows = int(crossing.sum())
    n_groups = int(df.loc[crossing, GROUP_COL].nunique())
    return df[~crossing].copy(), n_rows, n_groups


def _derive_training_subsets(train_full: pd.DataFrame
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _cap_per_issue(g):
        n = min(len(g), TRAIN_CAP_PER_ISSUE)
        return g.sample(n=n, random_state=RANDOM_SEED) if n < len(g) else g

    train_cap_issue = (train_full
                       .groupby(ISSUE_COL, group_keys=False)[train_full.columns.tolist()]
                       .apply(_cap_per_issue)
                       .reset_index(drop=True))

    train_3k_team = (train_full
                     .sample(frac=1, random_state=RANDOM_SEED)
                     .groupby(TEAM_COL, group_keys=False)
                     .head(BASELINE2_CAP_PER_TEAM)
                     .reset_index(drop=True))

    return train_cap_issue, train_3k_team


def _assert_invariants(train_full, val, test):
    assert len(train_full) == 774_412, f"train_full={len(train_full):,}"
    assert len(val) == 153_502, f"val={len(val):,}"
    assert len(test) == 109_834, f"test={len(test):,}"
    g_tr, g_va, g_te = (set(df[GROUP_COL]) for df in (train_full, val, test))
    assert g_tr.isdisjoint(g_va), "train & val share groups"
    assert g_tr.isdisjoint(g_te), "train & test share groups"
    assert g_va.isdisjoint(g_te), "val & test share groups"
    for name, df in (("train_full", train_full), ("val", val), ("test", test)):
        n_teams = df[TEAM_COL].nunique()
        assert n_teams == 11, f"{name} has {n_teams} teams (want 11)"


def _assert_parity_with_baseline2(test: pd.DataFrame) -> bool:
    b2 = pd.read_csv(BASELINE2_METRICS / "predictions_test.csv",
                     usecols=[ID_COL], dtype={ID_COL: "string"})
    ours = set(test[ID_COL].astype(str))
    theirs = set(b2[ID_COL].astype(str))
    if ours != theirs:
        only_ours = len(ours - theirs)
        only_theirs = len(theirs - ours)
        raise AssertionError(
            f"test IDs differ: only in ours={only_ours:,}, "
            f"only in baseline 2={only_theirs:,}")
    return True


def _per_label_counts(df, col):
    return df[col].value_counts().sort_index().to_dict()


def build_all(verbose: bool = True) -> dict:
    csv_to_parquet(verbose=verbose)

    if verbose:
        print(f"[read] {FULL_PARQUET}")
    df = pd.read_parquet(FULL_PARQUET)
    df[ID_COL] = df[ID_COL].astype("string")
    df = df[df[TEAM_COL].astype(str).str.len() > 0].copy()
    df[ISSUE_COL] = df["product_issue_id"].map(canonical_issue)

    df = _assign_periods(df)
    df, n_crossing_rows, n_crossing_groups = _drop_crossing_groups(df)
    if verbose:
        print(f"[split] dropped {n_crossing_rows:,} rows across "
              f"{n_crossing_groups:,} crossing groups")

    train_full = df[df["_split"] == "train"].drop(columns="_split").reset_index(drop=True)
    val = df[df["_split"] == "val"].drop(columns="_split").reset_index(drop=True)
    test = df[df["_split"] == "test"].drop(columns="_split").reset_index(drop=True)

    _assert_invariants(train_full, val, test)
    parity_ok = _assert_parity_with_baseline2(test)

    train_cap_issue, train_3k_team = _derive_training_subsets(train_full)

    SPLITS_V2.mkdir(parents=True, exist_ok=True)
    for name, part in (
        ("train_full", train_full),
        ("train_cap_issue", train_cap_issue),
        ("train_3k_team", train_3k_team),
        ("val", val),
        ("test", test),
    ):
        part[_OUT_COLS].to_parquet(SPLITS_V2 / f"{name}.parquet", index=False)
    train_full[[ID_COL]].to_parquet(SPLITS_V2 / "train_full_ids.parquet", index=False)

    report = {
        "counts": {
            "train_full": int(len(train_full)),
            "train_cap_issue": int(len(train_cap_issue)),
            "train_3k_team": int(len(train_3k_team)),
            "val": int(len(val)),
            "test": int(len(test)),
        },
        "dropped_crossing_group_rows": int(n_crossing_rows),
        "dropped_crossing_groups": int(n_crossing_groups),
        "per_team": {
            name: _per_label_counts(part, TEAM_COL)
            for name, part in (("train_full", train_full),
                               ("val", val), ("test", test))
        },
        "per_issue_sizes": {
            name: {"min": int(part[ISSUE_COL].value_counts().min()),
                   "n_issues": int(part[ISSUE_COL].nunique())}
            for name, part in (("train_full", train_full),
                               ("val", val), ("test", test))
        },
        "date_ranges": {
            name: {"min": str(part[DATE_COL].min()),
                   "max": str(part[DATE_COL].max())}
            for name, part in (("train_full", train_full),
                               ("val", val), ("test", test))
        },
        "parity": {"test_ids_equal_baseline2": bool(parity_ok)},
        "seed": RANDOM_SEED,
    }
    (SPLITS_V2 / "split_report.json").write_text(json.dumps(report, indent=2))
    tracked = METRICS_ROOT / "improved_model_1"
    tracked.mkdir(parents=True, exist_ok=True)
    (tracked / "split_report.json").write_text(json.dumps(report, indent=2))

    if verbose:
        for k, v in report["counts"].items():
            print(f"  {k:<16} {v:>8,}")
        print(f"[parity] test IDs match baseline 2 ({len(test):,} rows)")
        print(f"[write] {SPLITS_V2}")
    return report


if __name__ == "__main__":
    build_all(verbose=True)
