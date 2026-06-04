import argparse
import json
import sqlite3
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import DATABASE_PATH, REPORTS_DIR, TICKETS_TABLE_NAME


RANDOM_STATE = 42
TEST_SIZE = 0.2
TEXT_COLUMN = "text"

ALLOWED_TARGETS = ["priority", "queue"]
ALL_TARGETS = ["priority", "queue"]


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


def build_models(include_random_forest: bool = False) -> dict:
    models = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LinearSVC": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    if include_random_forest:
        models["RandomForestClassifier"] = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )

    return models


def build_pipeline(classifier) -> Pipeline:
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
            ("classifier", classifier),
        ]
    )

    return pipeline


def calculate_metrics(y_test, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }


def save_classification_report(
    target_column: str,
    model_name: str,
    y_test,
    y_pred,
) -> None:
    report = classification_report(y_test, y_pred, zero_division=0)

    output_path = (
        REPORTS_DIR / f"{target_column}_{model_name}_classification_report.txt"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report)


def train_and_evaluate_models(
    target_column: str,
    include_random_forest: bool = False,
) -> pd.DataFrame:
    df = load_data_from_database(target_column)

    print("=" * 80)
    print(f"Target: {target_column}")
    print("=" * 80)
    print("Dataset shape:", df.shape)
    print()
    print("Target distribution:")
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

    models = build_models(include_random_forest=include_random_forest)

    results = []

    for model_name, classifier in models.items():
        print(f"Training model: {model_name}")

        pipeline = build_pipeline(classifier)

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = pipeline.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred)
        metrics["target"] = target_column
        metrics["model"] = model_name
        metrics["training_time_seconds"] = round(training_time, 2)

        results.append(metrics)

        save_classification_report(
            target_column=target_column,
            model_name=model_name,
            y_test=y_test,
            y_pred=y_pred,
        )

        print(f"Finished: {model_name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 macro: {metrics['f1_macro']:.4f}")
        print(f"F1 weighted: {metrics['f1_weighted']:.4f}")
        print(f"Training time: {metrics['training_time_seconds']} s")
        print()

    results_df = pd.DataFrame(results)

    metric_columns = [
        "target",
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "training_time_seconds",
    ]

    results_df = results_df[metric_columns]
    results_df = results_df.sort_values(by="f1_macro", ascending=False)

    output_path = REPORTS_DIR / f"{target_column}_model_comparison.csv"
    results_df.to_csv(output_path, index=False)

    json_output_path = REPORTS_DIR / f"{target_column}_model_comparison.json"

    with open(json_output_path, "w", encoding="utf-8") as file:
        json.dump(results_df.to_dict(orient="records"), file, indent=4)

    print(f"Model comparison saved to: {output_path}")
    print(f"Model comparison JSON saved to: {json_output_path}")
    print()
    print("Model comparison:")
    print(results_df)
    print()

    best_model = results_df.iloc[0]

    print("Best model:")
    print(f"Target: {best_model['target']}")
    print(f"Model: {best_model['model']}")
    print(f"F1 macro: {best_model['f1_macro']:.4f}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print()

    return results_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare text classification models for support ticket data."
    )

    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["all"] + ALLOWED_TARGETS,
        help="Target column to predict: priority, queue or all.",
    )

    parser.add_argument(
        "--include-random-forest",
        action="store_true",
        help="Include RandomForestClassifier in comparison. It can be slower for TF-IDF data.",
    )

    return parser.parse_args()


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    args = parse_args()

    if args.target == "all":
        targets = ALL_TARGETS
    else:
        targets = [args.target]

    all_results = []

    for target in targets:
        result = train_and_evaluate_models(
            target_column=target,
            include_random_forest=args.include_random_forest,
        )
        all_results.append(result)

    combined_results = pd.concat(all_results, ignore_index=True)

    combined_output_path = REPORTS_DIR / "all_targets_model_comparison.csv"
    combined_results.to_csv(combined_output_path, index=False)

    print("=" * 80)
    print("Combined model comparison:")
    print(combined_results)
    print()
    print(f"Combined comparison saved to: {combined_output_path}")


if __name__ == "__main__":
    main()
