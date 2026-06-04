import sqlite3

import pandas as pd

from src.config import DATABASE_PATH


def main():
    query = """
    SELECT
        prediction_id,
        subject,
        predicted_priority,
        predicted_queue,
        priority_score,
        queue_score,
        model_version,
        created_at
    FROM predictions
    ORDER BY prediction_id DESC
    LIMIT 10;
    """

    with sqlite3.connect(DATABASE_PATH) as connection:
        df = pd.read_sql_query(query, connection)

    print("Last 10 predictions:")
    print(df)


if __name__ == "__main__":
    main()
