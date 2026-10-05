"""Hyperparameter grid and artifact directory for improved_model_1."""
from src.utils.config import METRICS_ROOT

ARTIFACT_DIR = METRICS_ROOT / "improved_model_1"

USE_CHAR_DEFAULT = True
WORD_MAX_FEATURES = 150_000
CHAR_MAX_FEATURES = 150_000

C_TEAM_GRID = (0.1, 0.25, 0.5, 1.0)
C_ISSUE_GRID = (0.1, 0.25, 0.5, 1.0)

MIN_ISSUE_ROWS_ISOTONIC = 200
MIN_ISSUES_FOR_MULTICLASS = 2

MAX_ARTIFACT_MB = 95
