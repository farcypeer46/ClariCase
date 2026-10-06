import pandas as pd

from src.utils.registry import COLUMNS, log_result, seed_from_baseline_2


def test_log_result_appends_row(tmp_path, monkeypatch):
    csv = tmp_path / "model_comparison.csv"
    monkeypatch.setattr("src.utils.registry.COMPARISON_CSV", csv)
    log_result("test_model", "test", {"team_acc": 0.5, "team_macro_f1": 0.4,
                                       "n_eval": 10})
    df = pd.read_csv(csv)
    assert list(df.columns) == COLUMNS
    assert df.iloc[0]["model"] == "test_model"


def test_seed_row_is_present_after_seeding(tmp_path, monkeypatch):
    csv = tmp_path / "model_comparison.csv"
    monkeypatch.setattr("src.utils.registry.COMPARISON_CSV", csv)
    seed_from_baseline_2()
    df = pd.read_csv(csv)
    assert (df["model"] == "baseline_model_2").any()
