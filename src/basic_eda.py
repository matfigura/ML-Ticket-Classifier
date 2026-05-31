import sqlite3

import matplotlib.pyplot as plt
import pandas as pd

from src.config import DATABASE_PATH, REPORTS_DIR, TICKETS_TABLE_NAME


def load_tickets_from_database() -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM {TICKETS_TABLE_NAME};
    """

    with sqlite3.connect(DATABASE_PATH) as connection:
        df = pd.read_sql_query(query, connection)

    return df


def save_distribution_report(df: pd.DataFrame, column: str, output_name: str) -> pd.DataFrame:
    distribution = (
        df[column]
        .value_counts(dropna=False)
        .reset_index()
    )

    distribution.columns = [column, "tickets_count"]
    distribution["percentage"] = (
        distribution["tickets_count"] / len(df) * 100
    ).round(2)

    output_path = REPORTS_DIR / output_name
    distribution.to_csv(output_path, index=False)

    return distribution


def save_missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    missing_values = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percentage": (df.isna().sum().values / len(df) * 100).round(2),
    })

    output_path = REPORTS_DIR / "missing_values.csv"
    missing_values.to_csv(output_path, index=False)

    return missing_values


def save_text_length_report(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["text_length_chars"] = df["text"].astype(str).str.len()
    df["text_length_words"] = df["text"].astype(str).str.split().str.len()

    text_length_summary = df[
        ["text_length_chars", "text_length_words"]
    ].describe().round(2)

    output_path = REPORTS_DIR / "text_length_summary.csv"
    text_length_summary.to_csv(output_path)

    return df


def save_duplicates_report(df: pd.DataFrame) -> pd.DataFrame:
    duplicated_text_count = df.duplicated(subset=["text"]).sum()
    duplicated_subject_body_count = df.duplicated(subset=["subject", "body"]).sum()

    duplicates_report = pd.DataFrame({
        "metric": [
            "duplicated_text_count",
            "duplicated_subject_body_count",
        ],
        "value": [
            duplicated_text_count,
            duplicated_subject_body_count,
        ],
    })

    output_path = REPORTS_DIR / "duplicates_report.csv"
    duplicates_report.to_csv(output_path, index=False)

    return duplicates_report


def plot_bar_distribution(distribution: pd.DataFrame, label_column: str, title: str, output_name: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(distribution[label_column].astype(str), distribution["tickets_count"])
    plt.title(title)
    plt.xlabel(label_column)
    plt.ylabel("Number of tickets")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = REPORTS_DIR / output_name
    plt.savefig(output_path)
    plt.close()


def plot_text_length_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(df["text_length_words"], bins=50)
    plt.title("Ticket text length distribution")
    plt.xlabel("Number of words")
    plt.ylabel("Number of tickets")
    plt.tight_layout()

    output_path = REPORTS_DIR / "text_length_distribution.png"
    plt.savefig(output_path)
    plt.close()


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_tickets_from_database()

    print("Dataset shape:")
    print(df.shape)
    print()

    print("Columns:")
    print(df.columns.tolist())
    print()

    priority_distribution = save_distribution_report(
        df=df,
        column="priority",
        output_name="priority_distribution.csv",
    )

    print("Priority distribution:")
    print(priority_distribution)
    print()

    type_distribution = save_distribution_report(
        df=df,
        column="type",
        output_name="type_distribution.csv",
    )

    print("Type distribution:")
    print(type_distribution)
    print()

    queue_distribution = save_distribution_report(
        df=df,
        column="queue",
        output_name="queue_distribution.csv",
    )

    print("Top 10 queues:")
    print(queue_distribution.head(10))
    print()

    missing_values = save_missing_values_report(df)

    print("Missing values:")
    print(missing_values)
    print()

    duplicates_report = save_duplicates_report(df)

    print("Duplicates report:")
    print(duplicates_report)
    print()

    df_with_lengths = save_text_length_report(df)

    print("Text length summary:")
    print(df_with_lengths[["text_length_chars", "text_length_words"]].describe().round(2))
    print()

    plot_bar_distribution(
        distribution=priority_distribution,
        label_column="priority",
        title="Ticket priority distribution",
        output_name="priority_distribution.png",
    )

    plot_bar_distribution(
        distribution=type_distribution,
        label_column="type",
        title="Ticket type distribution",
        output_name="type_distribution.png",
    )

    plot_bar_distribution(
        distribution=queue_distribution.head(10),
        label_column="queue",
        title="Top 10 ticket queues",
        output_name="top_10_queues.png",
    )

    plot_text_length_distribution(df_with_lengths)

    print(f"EDA reports saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()