import json
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import DATABASE_PATH, MODELS_DIR, REPORTS_DIR, TICKETS_TABLE_NAME


RANDOM_STATE = 42
TEXT_COLUMN = "text"

FINAL_TARGETS = ["priority", "queue"]


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


def build_final_pipeline() -> Pipeline:
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
                LinearSVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return pipeline


def train_and_save_model(target_column: str) -> dict:
    df = load_data_from_database(target_column)

    X = df[TEXT_COLUMN]
    y = df[target_column]

    print("=" * 80)
    print(f"Training final model for target: {target_column}")
    print("=" * 80)
    print(f"Dataset shape: {df.shape}")
    print()
    print("Target distribution:")
    print(y.value_counts())
    print()

    model = build_final_pipeline()

    print("Training final model on full dataset...")
    model.fit(X, y)
    print("Training completed.")
    print()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{target_column}_model.joblib"
    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")
    print()

    model_metadata = {
        "target": target_column,
        "model_path": str(model_path),
        "algorithm": "LinearSVC",
        "text_vectorizer": "TfidfVectorizer",
        "number_of_samples": int(len(df)),
        "number_of_classes": int(y.nunique()),
        "classes": sorted(y.unique().tolist()),
        "tfidf_params": {
            "lowercase": True,
            "stop_words": "english",
            "max_features": 10000,
            "ngram_range": [1, 2],
            "min_df": 2,
        },
        "classifier_params": {
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    return model_metadata


def save_models_metadata(metadata: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "final_models_metadata.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"Final models metadata saved to: {output_path}")


def main():
    all_metadata = []

    for target in FINAL_TARGETS:
        metadata = train_and_save_model(target)
        all_metadata.append(metadata)

    save_models_metadata(all_metadata)

    print("=" * 80)
    print("Final models training completed.")
    print("=" * 80)

    for metadata in all_metadata:
        print(
            f"{metadata['target']}: "
            f"{metadata['algorithm']} model saved to {metadata['model_path']}"
        )


if __name__ == "__main__":
    main()
