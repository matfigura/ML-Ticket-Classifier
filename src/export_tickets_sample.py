import sqlite3

import pandas as pd

from src.config import DATABASE_PATH, PROCESSED_DATA_DIR, TICKETS_TABLE_NAME


def export_sample(limit: int = 1000) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    query = f"""
    SELECT
        ticket_id,
        subject,
        body,
        text,
        type,
        queue,
        priority,
        language
    FROM {TICKETS_TABLE_NAME}
    LIMIT {limit};
    """

    with sqlite3.connect(DATABASE_PATH) as connection:
        df = pd.read_sql_query(query, connection)

    output_path = PROCESSED_DATA_DIR / "tickets_sample.csv"
    df.to_csv(output_path, index=False)

    print(f"Exported {len(df)} tickets to: {output_path}")


def main():
    export_sample(limit=1000)


if __name__ == "__main__":
    main()