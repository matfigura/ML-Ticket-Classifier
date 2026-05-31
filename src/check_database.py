import sqlite3

import pandas as pd

from src.config import DATABASE_PATH, TICKETS_TABLE_NAME


def run_query(query: str) -> pd.DataFrame:
    with sqlite3.connect(DATABASE_PATH) as connection:
        return pd.read_sql_query(query, connection)


def main():
    print(f"Database path: {DATABASE_PATH}")
    print(f"Table name: {TICKETS_TABLE_NAME}")
    print()

    total_tickets = run_query(
        f"""
        SELECT COUNT(*) AS total_tickets
        FROM {TICKETS_TABLE_NAME};
        """
    )

    print("Total tickets:")
    print(total_tickets)
    print()

    tickets_by_priority = run_query(
        f"""
        SELECT priority, COUNT(*) AS tickets_count
        FROM {TICKETS_TABLE_NAME}
        GROUP BY priority
        ORDER BY tickets_count DESC;
        """
    )

    print("Tickets by priority:")
    print(tickets_by_priority)
    print()

    tickets_by_queue = run_query(
        f"""
        SELECT queue, COUNT(*) AS tickets_count
        FROM {TICKETS_TABLE_NAME}
        GROUP BY queue
        ORDER BY tickets_count DESC
        LIMIT 10;
        """
    )

    print("Top 10 queues:")
    print(tickets_by_queue)
    print()

    tickets_by_type = run_query(
        f"""
        SELECT type, COUNT(*) AS tickets_count
        FROM {TICKETS_TABLE_NAME}
        GROUP BY type
        ORDER BY tickets_count DESC;
        """
    )

    print("Tickets by type:")
    print(tickets_by_type)
    print()

    sample_tickets = run_query(
        f"""
        SELECT ticket_id, subject, priority, queue, type, language
        FROM {TICKETS_TABLE_NAME}
        LIMIT 5;
        """
    )

    print("Sample tickets:")
    print(sample_tickets)


if __name__ == "__main__":
    main()