import argparse
import random
import sqlite3
from datetime import datetime

import joblib

from src.config import DATABASE_PATH, MODELS_DIR


PRIORITY_MODEL_PATH = MODELS_DIR / "priority_model.joblib"
QUEUE_MODEL_PATH = MODELS_DIR / "queue_model.joblib"

MODEL_VERSION = "tfidf_linearsvc_v1"


SAMPLE_TICKETS = [
    {
        "subject": "Cannot access my account",
        "body": "I cannot log in after resetting my password. Please help me regain access.",
    },
    {
        "subject": "Invoice issue",
        "body": "I was charged twice for my last invoice and need help with the payment.",
    },
    {
        "subject": "Product return request",
        "body": "I received a damaged product and would like to return it or exchange it.",
    },
    {
        "subject": "VPN connection problem",
        "body": "The VPN connection fails every time I try to connect from home.",
    },
    {
        "subject": "Request for software installation",
        "body": "Please install the latest version of the design software on my company laptop.",
    },
    {
        "subject": "Printer is not working",
        "body": "The office printer does not print any documents and shows a paper jam error.",
    },
    {
        "subject": "Payment confirmation needed",
        "body": "I made a payment yesterday but it is still not visible on my account.",
    },
    {
        "subject": "Service outage",
        "body": "The application is unavailable for all users in our department since this morning.",
    },
    {
        "subject": "Password reset request",
        "body": "I forgot my password and need a reset link to access the platform.",
    },
    {
        "subject": "Question about product features",
        "body": "Could you explain whether the premium plan includes advanced reporting features?",
    },
    {
        "subject": "Change billing address",
        "body": "Please update the billing address for our company account before the next invoice.",
    },
    {
        "subject": "Refund request",
        "body": "I would like to request a refund because the product does not meet my expectations.",
    },
    {
        "subject": "New employee account",
        "body": "Please create an account and email access for a new employee starting next Monday.",
    },
    {
        "subject": "Application crashes",
        "body": "The application crashes every time I try to upload a large file.",
    },
    {
        "subject": "Demo request",
        "body": "I would like to schedule a product demo for our sales team next week.",
    },
    {
        "subject": "Order status question",
        "body": "Can you check the current status of my order? It has not arrived yet.",
    },
    {
        "subject": "Laptop overheating",
        "body": "My company laptop becomes very hot and shuts down during normal work.",
    },
    {
        "subject": "Email not syncing",
        "body": "My email inbox is not syncing on my mobile device or desktop client.",
    },
    {
        "subject": "System maintenance question",
        "body": "Will the system be unavailable during the planned maintenance window this weekend?",
    },
    {
        "subject": "Human resources document request",
        "body": "I need a copy of my employment confirmation document for administrative purposes.",
    },
    {
        "subject": "Product damaged during delivery",
        "body": "The package arrived damaged and one of the ordered items is broken.",
    },
    {
        "subject": "Cannot complete checkout",
        "body": "The checkout page displays an error when I try to complete the payment.",
    },
    {
        "subject": "Request to change subscription plan",
        "body": "Please change our subscription from the basic plan to the enterprise plan.",
    },
    {
        "subject": "Slow application performance",
        "body": "The application is extremely slow today and many users are reporting delays.",
    },
    {
        "subject": "General product inquiry",
        "body": "I have a general question about available product options and pricing.",
    },
]


def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run: python -m src.train_final_models"
        )

    return joblib.load(model_path)


def build_ticket_text(subject: str, body: str) -> str:
    subject = subject.strip()
    body = body.strip()

    return f"{subject} {body}".strip()


def predict_with_score(model, text: str, top_n: int = 3) -> tuple[str, float | None, list[dict]]:
    prediction = model.predict([text])[0]

    score = None
    top_predictions = []

    if hasattr(model, "decision_function"):
        decision_scores = model.decision_function([text])

        classifier = model.named_steps.get("classifier")
        classes = getattr(classifier, "classes_", None)

        if classes is not None and len(decision_scores.shape) == 2:
            class_scores = list(zip(classes, decision_scores[0]))
            class_scores = sorted(
                class_scores,
                key=lambda item: item[1],
                reverse=True,
            )

            top_predictions = [
                {
                    "class": class_name,
                    "score": float(class_score),
                }
                for class_name, class_score in class_scores[:top_n]
            ]

            scores_for_classes = dict(class_scores)
            score = float(scores_for_classes[prediction])

        elif classes is not None and len(decision_scores.shape) == 1:
            decision_value = float(decision_scores[0])
            score = abs(decision_value)

            top_predictions = [
                {
                    "class": prediction,
                    "score": score,
                }
            ]

    return prediction, score, top_predictions


def create_predictions_table_if_not_exists() -> None:
    create_table_query = """
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT NOT NULL,
        body TEXT NOT NULL,
        text TEXT NOT NULL,
        predicted_priority TEXT NOT NULL,
        predicted_queue TEXT NOT NULL,
        priority_score REAL,
        queue_score REAL,
        model_version TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(create_table_query)


def save_prediction_to_database(
    subject: str,
    body: str,
    text: str,
    predicted_priority: str,
    predicted_queue: str,
    priority_score: float | None,
    queue_score: float | None,
) -> int:
    create_predictions_table_if_not_exists()

    insert_query = """
    INSERT INTO predictions (
        subject,
        body,
        text,
        predicted_priority,
        predicted_queue,
        priority_score,
        queue_score,
        model_version,
        created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    created_at = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.execute(
            insert_query,
            (
                subject,
                body,
                text,
                predicted_priority,
                predicted_queue,
                priority_score,
                queue_score,
                MODEL_VERSION,
                created_at,
            ),
        )

        prediction_id = cursor.lastrowid

    return prediction_id


def predict_ticket(subject: str, body: str, save_to_database: bool = True) -> dict:
    text = build_ticket_text(subject, body)

    if not text:
        raise ValueError("Ticket text cannot be empty.")

    priority_model = load_model(PRIORITY_MODEL_PATH)
    queue_model = load_model(QUEUE_MODEL_PATH)

    predicted_priority, priority_score, priority_top_predictions = predict_with_score(
        priority_model,
        text,
    )

    predicted_queue, queue_score, queue_top_predictions = predict_with_score(
        queue_model,
        text,
    )

    result = {
        "subject": subject,
        "body": body,
        "text": text,
        "predicted_priority": predicted_priority,
        "predicted_queue": predicted_queue,
        "priority_score": priority_score,
        "queue_score": queue_score,
        "priority_top_predictions": priority_top_predictions,
        "queue_top_predictions": queue_top_predictions,
        "model_version": MODEL_VERSION,
        "prediction_id": None,
    }

    if save_to_database:
        prediction_id = save_prediction_to_database(
            subject=subject,
            body=body,
            text=text,
            predicted_priority=predicted_priority,
            predicted_queue=predicted_queue,
            priority_score=priority_score,
            queue_score=queue_score,
        )

        result["prediction_id"] = prediction_id

    return result


def get_int_choice(prompt: str, min_value: int, max_value: int) -> int:
    while True:
        user_input = input(prompt).strip()

        try:
            choice = int(user_input)
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if min_value <= choice <= max_value:
            return choice

        print(f"Invalid choice. Please enter a number from {min_value} to {max_value}.")


def get_non_empty_input(prompt: str) -> str:
    while True:
        value = input(prompt).strip()

        if value:
            return value

        print("This field cannot be empty.")


def read_custom_ticket() -> tuple[str, str]:
    subject = get_non_empty_input("Ticket subject: ")
    body = get_non_empty_input("Ticket body: ")

    return subject, body


def display_sample_tickets() -> None:
    print()
    print("=" * 80)
    print("Sample tickets")
    print("=" * 80)
    print("0. Random sample ticket")

    for index, ticket in enumerate(SAMPLE_TICKETS, start=1):
        print(f"{index}. {ticket['subject']}")


def choose_sample_ticket() -> tuple[str, str]:
    display_sample_tickets()

    choice = get_int_choice(
        prompt="Choose ticket number: ",
        min_value=0,
        max_value=len(SAMPLE_TICKETS),
    )

    if choice == 0:
        selected_ticket = random.choice(SAMPLE_TICKETS)
        print()
        print("Random ticket selected:")
        print(f"Subject: {selected_ticket['subject']}")
    else:
        selected_ticket = SAMPLE_TICKETS[choice - 1]

    return selected_ticket["subject"], selected_ticket["body"]


def choose_ticket_source() -> tuple[str, str]:
    print()
    print("=" * 80)
    print("ML Ticket Classifier")
    print("=" * 80)
    print("1. Enter your own ticket")
    print("2. Choose from sample tickets")

    choice = get_int_choice(
        prompt="Choose option: ",
        min_value=1,
        max_value=2,
    )

    if choice == 1:
        return read_custom_ticket()

    return choose_sample_ticket()


def display_prediction_result(result: dict) -> None:
    print()
    print("=" * 80)
    print("Ticket prediction result")
    print("=" * 80)
    print(f"Subject: {result['subject']}")
    print(f"Body: {result['body']}")
    print()
    print(f"Predicted priority: {result['predicted_priority']}")
    print(f"Predicted queue: {result['predicted_queue']}")
    print()
    print("Model details:")
    print(f"Model version: {result['model_version']}")
    print(f"Priority score: {result['priority_score']}")
    print(f"Queue score: {result['queue_score']}")

    print()
    print("Top priority candidates:")
    for index, candidate in enumerate(result["priority_top_predictions"], start=1):
        print(f"{index}. {candidate['class']} | score: {candidate['score']:.4f}")

    print()
    print("Top queue candidates:")
    for index, candidate in enumerate(result["queue_top_predictions"], start=1):
        print(f"{index}. {candidate['class']} | score: {candidate['score']:.4f}")

    if result["prediction_id"] is not None:
        print()
        print(f"Prediction saved to database with ID: {result['prediction_id']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict support ticket priority and queue."
    )

    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Ticket subject.",
    )

    parser.add_argument(
        "--body",
        type=str,
        default=None,
        help="Ticket body.",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save prediction to SQLite database.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.subject is not None or args.body is not None:
        subject = args.subject or get_non_empty_input("Ticket subject: ")
        body = args.body or get_non_empty_input("Ticket body: ")
    else:
        subject, body = choose_ticket_source()

    result = predict_ticket(
        subject=subject,
        body=body,
        save_to_database=not args.no_save,
    )

    display_prediction_result(result)


if __name__ == "__main__":
    main()