from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_predict_endpoint_returns_expected_invoice_prediction():
    payload = {
        "subject": "Invoice issue",
        "body": "I was charged twice for my last invoice and need help with the payment.",
        "save_to_database": True,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["subject"] == payload["subject"]
    assert data["body"] == payload["body"]

    assert data["predicted_priority"] == "high"
    assert data["predicted_queue"] == "billing and payments"

    assert isinstance(data["priority_score"], float)
    assert isinstance(data["queue_score"], float)

    assert data["priority_top_predictions"][0]["class_name"] == "high"
    assert data["queue_top_predictions"][0]["class_name"] == "billing and payments"

    assert data["model_version"] == "tfidf_linearsvc_v1"

    assert "prediction_id" in data
    assert isinstance(data["prediction_id"], int)
