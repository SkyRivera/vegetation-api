# tests/test_predictions.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_no_image():
    """Test predicción sin imagen"""
    response = client.post("/api/v1/vegetation/predict")
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_invalid_file():
    """Test con archivo inválido"""
    files = {"image": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/api/v1/vegetation/predict", files=files)
    assert response.status_code == 400


def test_model_info():
    """Test información del modelo"""
    response = client.get("/api/v1/model")
    if response.status_code == 503:
        pytest.skip("Modelo no cargado")
    else:
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
