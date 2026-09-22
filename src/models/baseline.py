"""Train and evaluate a TF-IDF baseline for complaint classification.

Run: python -m src.models.baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data.config import DATE, LABEL, ROOT, TEXT
from src.data.prepare import get_splits


def build_model(max_features: int = 75_000) -> Pipeline:
    """Create a sparse text-classification pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LinearSVC(class_weight="balanced")),
        ]
    )


def metric_summary(y_true: pd.Series, y_pred: object) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def train_and_evaluate(output_dir: Path, max_features: int) -> None:
    train, test = get_splits(verbose=True)
    x_train, y_train = train[TEXT], train[LABEL]
    x_test, y_test = test[TEXT], test[LABEL]

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train.to_frame(), y_train)
    dummy_metrics = metric_summary(y_test, dummy.predict(x_test.to_frame()))

    print("\n[train] fitting word (1,2)-gram TF-IDF + LinearSVC")
    model = build_model(max_features=max_features)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    model_metrics = metric_summary(y_test, predictions)

    labels = sorted(set(y_train) | set(y_test))
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps({"dummy": dummy_metrics, "model": model_metrics}, indent=2)
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2)
    )
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix.csv", index_label="actual\\predicted"
    )

    prediction_columns = [DATE, "Complaint ID"]
    prediction_frame = test[prediction_columns].copy()
    prediction_frame["actual"] = y_test.astype(str)
    prediction_frame["predicted"] = predictions
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)

    print("\n[results]")
    print(f"  {'metric':<14} {'dummy':>9} {'model':>9}")
    for metric in model_metrics:
        print(
            f"  {metric:<14} {dummy_metrics[metric]:>9.3f} "
            f"{model_metrics[metric]:>9.3f}"
        )
    print(f"\n[write] artifacts saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "baseline",
        help="Directory for the model and evaluation artifacts.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=75_000,
        help="Maximum TF-IDF vocabulary size.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args.output_dir, args.max_features)
