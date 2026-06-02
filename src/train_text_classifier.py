import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import DATABASE_PATH, MODELS_DIR, REPORTS_DIR, TICKETS_TABLE_NAME


RANDOM_STATE = 42
TEST_SIZE = 0.2
TEXT_COLUMN = "text"

ALLOWED_TARGETS = ["priority", "queue"]


def load_data_from_database(target_column: str) -> pd.DataFrame:
    query = f"""
    SELECT
        ticket_id,
        text,
        {target_column}
    FROM {TICKETS_TABLE_NAME}
    WHERE text IS NOT NULL
      AND LENGTH(TRIM(text)) > 0
      AND {target_column} IS NOT NULL
      AND {target_column} != 'unknown';
    """

    with sqlite3.connect(DATABASE_PATH) as connection:
        df = pd.read_sql_query(query, connection)

    return df


def build_model_pipeline() -> Pipeline:
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return pipeline


def calculate_metrics(y_test, y_pred) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
    }

    return metrics


def save_metrics(metrics: dict, target_column: str) -> None:
    output_path = REPORTS_DIR / f"{target_column}_model_metrics.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"Metrics saved to: {output_path}")


def save_classification_report(y_test, y_pred, target_column: str) -> None:
    output_path = REPORTS_DIR / f"{target_column}_classification_report.txt"

    report = classification_report(y_test, y_pred, zero_division=0)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"Classification report saved to: {output_path}")
    print()
    print("Classification report:")
    print(report)


def save_confusion_matrix(y_test, y_pred, target_column: str) -> None:
    labels = sorted(y_test.unique())

    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )

    csv_output_path = REPORTS_DIR / f"{target_column}_confusion_matrix.csv"
    matrix_df.to_csv(csv_output_path)

    print(f"Confusion matrix CSV saved to: {csv_output_path}")

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(values_format="d")
    plt.title(f"{target_column.capitalize()} model confusion matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    image_output_path = REPORTS_DIR / f"{target_column}_confusion_matrix.png"
    plt.savefig(image_output_path)
    plt.close()

    print(f"Confusion matrix image saved to: {image_output_path}")


def save_model(model: Pipeline, target_column: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_output_path = MODELS_DIR / f"{target_column}_model.joblib"
    joblib.dump(model, model_output_path)

    print(f"Model saved to: {model_output_path}")


def train_model(target_column: str) -> None:
    if target_column not in ALLOWED_TARGETS:
        raise ValueError(
            f"Unsupported target: {target_column}. "
            f"Allowed targets: {ALLOWED_TARGETS}"
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data_from_database(target_column)

    print(f"Training model for target: {target_column}")
    print()
    print("Loaded dataset:")
    print(df.shape)
    print()

    print(f"{target_column.capitalize()} distribution:")
    print(df[target_column].value_counts())
    print()

    X = df[TEXT_COLUMN]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Train size:", X_train.shape[0])
    print("Test size:", X_test.shape[0])
    print()

    model = build_model_pipeline()

    print("Training model...")
    model.fit(X_train, y_train)
    print("Training completed.")
    print()

    y_pred = model.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred)

    print("Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print()

    save_metrics(metrics, target_column)
    save_classification_report(y_test, y_pred, target_column)
    save_confusion_matrix(y_test, y_pred, target_column)
    save_model(model, target_column)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train text classifier for support ticket data."
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=ALLOWED_TARGETS,
        help="Target column to predict.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train_model(target_column=args.target)


if __name__ == "__main__":
    main()