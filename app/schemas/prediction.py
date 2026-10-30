# app/schemas/prediction.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, List


# ---------- Submodelos tipados ----------

class ModelInfo(BaseModel):
    """Información del modelo que generó la predicción."""
    name: str
    version: str


class PredictionDetail(BaseModel):
    """
    Resultado principal de la predicción.

    - category: etiqueta discreta ('semi_humeda' | 'seca' | 'muy_seca' | 'muy_seca_extrema')
    - confidence: probabilidad de la clase ganadora [0.0 - 1.0]
    """
    category: Literal['semi_humeda', 'seca', 'muy_seca', 'muy_seca_extrema']
    confidence: float = Field(..., ge=0.0, le=1.0)


class DistributionDetail(BaseModel):
    """
    Distribución porcentual de probabilidad entre las 4 categorías.
    Cada valor está en [0.0 - 100.0].
    """
    semi_humeda:      float = Field(..., ge=0.0, le=100.0)
    seca:             float = Field(..., ge=0.0, le=100.0)
    muy_seca:         float = Field(..., ge=0.0, le=100.0)
    muy_seca_extrema: float = Field(..., ge=0.0, le=100.0)


class FeaturesDetail(BaseModel):
    """
    Características extraídas de la imagen.
    El orden coincide con el vector usado por el modelo (11 features).
    """
    green_pixel_ratio:      float
    brown_pixel_ratio:      float
    ndvi_simulated:         float
    gci:                    float
    hsv_mean_h:             float
    hsv_mean_s:             float
    hsv_mean_v:             float
    green_brown_ratio:      float
    green_minus_brown:      float
    ndvi_gci_interaction:   float
    hue_distance_to_green:  float


# ---------- Modelos de request/response ----------

class PredictionCreate(BaseModel):
    """Modelo interno para persistir una predicción en DB (no expuesto al cliente)."""
    prediction_id: str
    created_at: datetime
    model_version: str
    predicted_class: str
    confidence: float
    semi_humeda_percentage: float
    seca_percentage: float
    muy_seca_percentage: float
    muy_seca_extrema_percentage: float
    image_filename: str
    processing_time_ms: int
    # 11 features
    green_pixel_ratio: float
    brown_pixel_ratio: float
    ndvi_sim: float
    gci: float
    hsv_mean_h: float
    hsv_mean_s: float
    hsv_mean_v: float
    green_brown_ratio: float
    green_minus_brown: float
    ndvi_gci_interaction: float
    hue_distance_to_green: float


class PredictionResponse(BaseModel):
    """Respuesta al cliente tras clasificar una imagen."""
    prediction_id: str
    timestamp: datetime
    model: ModelInfo
    prediction: PredictionDetail
    distribution: DistributionDetail
    features: FeaturesDetail

    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2026-09-18T22:00:00Z",
                "model": {
                    "name": "vegetation_classifier",
                    "version": "v20260918_120000"
                },
                "prediction": {
                    "category": "seca",
                    "confidence": 0.87
                },
                "distribution": {
                    "semi_humeda": 12.2,
                    "seca": 66.3,
                    "muy_seca": 16.4,
                    "muy_seca_extrema": 5.1
                },
                "features": {
                    "green_pixel_ratio": 0.30,
                    "brown_pixel_ratio": 0.45,
                    "ndvi_simulated": 0.10,
                    "gci": 0.05,
                    "hsv_mean_h": 40.2,
                    "hsv_mean_s": 110.4,
                    "hsv_mean_v": 130.1,
                    "green_brown_ratio": 0.667,
                    "green_minus_brown": -0.15,
                    "ndvi_gci_interaction": 0.005,
                    "hue_distance_to_green": 0.33
                }
            }
        }
