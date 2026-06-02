import sqlite3
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import DATABASE_PATH
from src.predict import (
    PRIORITY_MODEL_PATH,
    QUEUE_MODEL_PATH,
    MODEL_VERSION,
    SAMPLE_TICKETS,
    build_ticket_text,
    load_model,
    predict_with_score,
    save_prediction_to_database,
)


app = FastAPI(
    title="ML Ticket Classifier API",
    description=(
        "REST API for predicting support ticket priority and queue "
        "using TF-IDF and LinearSVC models."
    ),
    version="1.0.0",
)


priority_model = None
queue_model = None


class TicketRequest(BaseModel):
    subject: str = Field(
        min_length=1,
        example="Invoice issue",
    )
    body: str = Field(
        min_length=1,
        example="I was charged twice for my last invoice and need help with the payment.",
    )
    save_to_database: bool = Field(
        default=True,
        description="Whether to save prediction result to SQLite database.",
    )


class CandidatePrediction(BaseModel):
    class_name: str
    score: float


class TicketPredictionResponse(BaseModel):
    subject: str
    body: str
    predicted_priority: str
    predicted_queue: str
    priority_score: Optional[float]
    queue_score: Optional[float]
    priority_top_predictions: list[CandidatePrediction]
    queue_top_predictions: list[CandidatePrediction]
    model_version: str
    prediction_id: Optional[int]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    priority_model_exists: bool
    queue_model_exists: bool


def get_models():
    global priority_model, queue_model

    if priority_model is None:
        priority_model = load_model(PRIORITY_MODEL_PATH)

    if queue_model is None:
        queue_model = load_model(QUEUE_MODEL_PATH)

    return priority_model, queue_model


def convert_top_predictions(top_predictions: list[dict]) -> list[CandidatePrediction]:
    return [
        CandidatePrediction(
            class_name=candidate["class"],
            score=candidate["score"],
        )
        for candidate in top_predictions
    ]


@app.get("/", tags=["General"])
def root():
    return {
        "message": "ML Ticket Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    return HealthResponse(
        status="ok",
        model_version=MODEL_VERSION,
        priority_model_exists=PRIORITY_MODEL_PATH.exists(),
        queue_model_exists=QUEUE_MODEL_PATH.exists(),
    )


@app.post(
    "/predict",
    response_model=TicketPredictionResponse,
    tags=["Prediction"],
)
def predict_ticket_api(ticket: TicketRequest):
    subject = ticket.subject.strip()
    body = ticket.body.strip()
    text = build_ticket_text(subject, body)

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Ticket subject and body cannot be empty.",
        )

    try:
        loaded_priority_model, loaded_queue_model = get_models()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error

    predicted_priority, priority_score, priority_top_predictions = predict_with_score(
        loaded_priority_model,
        text,
    )

    predicted_queue, queue_score, queue_top_predictions = predict_with_score(
        loaded_queue_model,
        text,
    )

    prediction_id = None

    if ticket.save_to_database:
        prediction_id = save_prediction_to_database(
            subject=subject,
            body=body,
            text=text,
            predicted_priority=predicted_priority,
            predicted_queue=predicted_queue,
            priority_score=priority_score,
            queue_score=queue_score,
        )

    return TicketPredictionResponse(
        subject=subject,
        body=body,
        predicted_priority=predicted_priority,
        predicted_queue=predicted_queue,
        priority_score=priority_score,
        queue_score=queue_score,
        priority_top_predictions=convert_top_predictions(priority_top_predictions),
        queue_top_predictions=convert_top_predictions(queue_top_predictions),
        model_version=MODEL_VERSION,
        prediction_id=prediction_id,
    )


@app.get("/sample-tickets", tags=["Samples"])
def get_sample_tickets():
    return {
        "count": len(SAMPLE_TICKETS),
        "sample_tickets": [
            {
                "id": index,
                "subject": ticket["subject"],
                "body": ticket["body"],
            }
            for index, ticket in enumerate(SAMPLE_TICKETS, start=1)
        ],
    }


@app.get("/sample-tickets/{ticket_id}", tags=["Samples"])
def get_sample_ticket(ticket_id: int):
    if ticket_id < 1 or ticket_id > len(SAMPLE_TICKETS):
        raise HTTPException(
            status_code=404,
            detail=f"Sample ticket not found. Choose ID from 1 to {len(SAMPLE_TICKETS)}.",
        )

    ticket = SAMPLE_TICKETS[ticket_id - 1]

    return {
        "id": ticket_id,
        "subject": ticket["subject"],
        "body": ticket["body"],
    }


@app.get("/predictions", tags=["Predictions"])
def get_last_predictions(
    limit: int = Query(default=10, ge=1, le=100),
):
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
    LIMIT ?;
    """

    try:
        with sqlite3.connect(DATABASE_PATH) as connection:
            cursor = connection.execute(query, (limit,))
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

    except sqlite3.OperationalError:
        return {
            "count": 0,
            "predictions": [],
            "message": "Predictions table does not exist yet.",
        }

    predictions = [
        dict(zip(columns, row))
        for row in rows
    ]

    return {
        "count": len(predictions),
        "predictions": predictions,
    }