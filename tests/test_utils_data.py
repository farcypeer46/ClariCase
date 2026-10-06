import pytest

from src.utils import data as D
from src.utils.config import ID_COL


@pytest.mark.slow
def test_load_split_returns_expected_columns():
    df = D.load_split("val")
    expected = {"complaint_id", "date_received", "company", "complaint_text",
                "team_id", "team_name", "issue_id", "leakage_group_id"}
    assert expected.issubset(df.columns)


@pytest.mark.slow
def test_load_holdout_has_issue_id():
    df = D.load_holdout_2026()
    assert "issue_id" in df.columns
    assert len(df) == 29_798
