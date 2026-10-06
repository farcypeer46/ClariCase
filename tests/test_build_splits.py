"""End-to-end split builder tests. Slow: reads 1.9 GB CSV once, then Parquet."""
import json

import pandas as pd
import pytest

from src.utils import config as c


@pytest.mark.slow
def test_splits_match_official_partitions():
    from src.utils.build_splits import build_all
    build_all(verbose=False)
    report = json.loads((c.SPLITS_V2 / "split_report.json").read_text())
    counts = report["counts"]
    assert counts["train_full"] == 774_412
    assert counts["val"] == 153_502
    assert counts["test"] == 109_834
    assert report["dropped_crossing_group_rows"] == 6_867
    assert report["dropped_crossing_groups"] == 1_949


@pytest.mark.slow
def test_test_ids_equal_baseline2():
    b2 = pd.read_csv(c.BASELINE2_METRICS / "predictions_test.csv",
                     usecols=[c.ID_COL], dtype={c.ID_COL: "string"})
    ours = pd.read_parquet(c.SPLITS_V2 / "test.parquet", columns=[c.ID_COL])
    assert set(ours[c.ID_COL].astype(str)) == set(b2[c.ID_COL].astype(str))
