"""Predict output contract test. Runs only if model.joblib exists."""
import pytest

from src.improved_model_1.config import ARTIFACT_DIR


@pytest.mark.skipif(not (ARTIFACT_DIR / "model.joblib").exists(),
                    reason="model.joblib not yet built")
def test_predict_output_shape():
    from src.improved_model_1.predict import predict
    out = predict("A collector keeps calling about a debt I already paid.")
    required = {"predicted_team_id", "predicted_team_name", "confidence",
                "top_k", "issue_top_k", "route", "model_name"}
    assert required.issubset(out)
    assert out["route"] in ("auto", "review")
    assert out["model_name"] == "improved_model_1"
    assert isinstance(out["top_k"], list) and len(out["top_k"]) > 0
    assert isinstance(out["issue_top_k"], list) and len(out["issue_top_k"]) > 0
