from fastapi.testclient import TestClient

from src.app.main import app


def test_predict_endpoint():
    client = TestClient(app)

    payload = {"series": [100.0 + i for i in range(14)]}  # same window_size

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "predicted_sales" in data
    assert isinstance(data["predicted_sales"], float)


def test_predict_by_category():
    client = TestClient(app)
    payload = {"series": [100.0 + i for i in range(14)]}
    response = client.post("/api/predict/Beauty", json=payload)
    assert response.status_code == 200
    assert "predicted_sales" in response.json()
