from src.utils import config as c


def test_paths_resolve_under_repo_root():
    assert c.ROOT.is_dir()
    assert c.FULL_CSV.name == "complaints_product_issues_2024_2025.csv"
    assert c.SPLITS_V2 == c.CACHE / "splits_v2"
    assert c.BASELINE2_METRICS == c.ROOT / "docs" / "metrics" / "baseline_model_2"


def test_temporal_boundaries_match_baseline_2():
    assert (c.TRAIN_END, c.VAL_END, c.TEST_END) == (
        "2025-07-01", "2025-10-01", "2026-01-01")


def test_seed_and_caps():
    assert c.RANDOM_SEED == 42
    assert c.TRAIN_CAP_PER_ISSUE == 10_000
    assert c.BASELINE2_CAP_PER_TEAM == 3_000
    assert c.MERGE_ISSUE_VARIANTS is True
    assert c.ISSUE_VARIANT_MERGES["T01_I007"] == "T01_I008"
    assert c.ISSUE_VARIANT_MERGES["T01_I003"] == "T01_I001"


def test_routing_constants_match_baseline_2():
    assert c.PRECISION_TARGETS == (0.95, 0.90, 0.85)
    assert c.TEAM_PRECISION_FLOOR == 0.85
    assert c.MIN_ROUTED == 50
    assert c.N_BOOTSTRAP == 1000
    assert c.ECE_BINS == 10
