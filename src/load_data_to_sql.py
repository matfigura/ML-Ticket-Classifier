import sqlite3

import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    DATABASE_PATH,
    TICKETS_TABLE_NAME,
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
)


def find_csv_file():
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {RAW_DATA_DIR}. "
            "Download the dataset and place it in data/raw/."
        )

    if len(csv_files) > 1:
        print("More than one CSV file found. Using the first one:")
        for file in csv_files:
            print(f"- {file.name}")

    return csv_files[0]


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    )
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns in dataset: " + ", ".join(missing_columns)
        )


def prepare_tickets_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_column_names(df)
    validate_required_columns(df)

    available_optional_columns = [
        column for column in OPTIONAL_COLUMNS if column in df.columns
    ]

    selected_columns = REQUIRED_COLUMNS + available_optional_columns
    df = df[selected_columns].copy()

    df["language"] = df["language"].astype(str).str.lower().str.strip()
    df = df[df["language"] == "en"].copy()

    df["subject"] = df["subject"].fillna("").astype(str).str.strip()
    df["body"] = df["body"].fillna("").astype(str).str.strip()

    df["text"] = (df["subject"] + " " + df["body"]).str.strip()

    df = df[df["text"].str.len() > 0].copy()

    df["type"] = df["type"].fillna("unknown").astype(str).str.lower().str.strip()
    df["queue"] = df["queue"].fillna("unknown").astype(str).str.lower().str.strip()
    df["priority"] = (
        df["priority"].fillna("unknown").astype(str).str.lower().str.strip()
    )

    df.insert(0, "ticket_id", range(1, len(df) + 1))

    return df


def save_to_sqlite(df: pd.DataFrame) -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        df.to_sql(
            TICKETS_TABLE_NAME,
            connection,
            if_exists="replace",
            index=False,
        )

        connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TICKETS_TABLE_NAME}_priority "
            f"ON {TICKETS_TABLE_NAME}(priority);"
        )

        connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TICKETS_TABLE_NAME}_queue "
            f"ON {TICKETS_TABLE_NAME}(queue);"
        )

        connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TICKETS_TABLE_NAME}_type "
            f"ON {TICKETS_TABLE_NAME}(type);"
        )


def main():
    csv_path = find_csv_file()

    print(f"Loading dataset from: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    print(f"Raw dataset shape: {raw_df.shape}")

    tickets_df = prepare_tickets_dataframe(raw_df)
    print(f"Prepared dataset shape: {tickets_df.shape}")

    save_to_sqlite(tickets_df)

    print(f"Data saved to SQLite database: {DATABASE_PATH}")
    print(f"Table name: {TICKETS_TABLE_NAME}")


if __name__ == "__main__":
    main()
