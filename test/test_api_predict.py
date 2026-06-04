from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_predict_endpoint_returns_valid_response_structure():
    payload = {
        "subject": "Invoice issue",
        "body": "I was charged twice for my last invoice and need help with the payment.",
        "save_to_database": False,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["subject"] == payload["subject"]
    assert data["body"] == payload["body"]

    assert "predicted_priority" in data
    assert "predicted_queue" in data
    assert "priority_score" in data
    assert "queue_score" in data
    assert "priority_top_predictions" in data
    assert "queue_top_predictions" in data
    assert "model_version" in data

    assert isinstance(data["predicted_priority"], str)
    assert isinstance(data["predicted_queue"], str)
    assert isinstance(data["priority_score"], float)
    assert isinstance(data["queue_score"], float)
    assert isinstance(data["priority_top_predictions"], list)
    assert isinstance(data["queue_top_predictions"], list)
    assert isinstance(data["model_version"], str)


def test_predict_endpoint_rejects_missing_body():
    payload = {
        "subject": "Invoice issue",
        "save_to_database": False,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422