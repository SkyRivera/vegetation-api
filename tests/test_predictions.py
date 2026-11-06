from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

client = TestClient(app)

def test_predict_no_image():
    response = client.post("/api/v1/vegetation/predict")
    assert response.status_code == 422

@patch("app.ml.model_loader.ModelLoader.is_loaded")
def test_predict_invalid_file(mock_is_loaded):
    """Test con archivo inválido"""
    # Forzamos a que la API crea que el modelo SÍ está cargado
    mock_is_loaded.return_value = True
    
    files = {"image": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/api/v1/vegetation/predict", files=files)
    
    # Ahora sí debería fallar por validación de imagen (400) y no por modelo (503)
    assert response.status_code == 400
    assert "Formato no soportado" in response.json()["detail"]

@patch("app.ml.model_loader.ModelLoader.is_loaded")
@patch("app.ml.model_loader.ModelLoader.get_model")
@patch("app.ml.model_loader.ModelLoader.get_scaler")
def test_model_info(mock_get_scaler, mock_get_model, mock_is_loaded):
    """Test información del modelo"""
    mock_is_loaded.return_value = True
    response = client.get("/api/v1/model")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "classes" in data