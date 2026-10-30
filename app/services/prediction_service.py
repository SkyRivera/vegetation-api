# app/services/prediction_service.py
import uuid
from datetime import datetime, timezone
from typing import Dict
import numpy as np
import logging

from app.ml.model_loader import ModelLoader
from app.ml.feature_extractor import FeatureExtractor
from app.db.repositories import PredictionRepository
from app.schemas.prediction import (
    PredictionCreate,
    PredictionResponse,
    PredictionDetail,
    DistributionDetail,
    FeaturesDetail,
    ModelInfo
)

logger = logging.getLogger(__name__)

# Categorías canónicas en orden estable
CATEGORIES = ['semi_humeda', 'seca', 'muy_seca', 'muy_seca_extrema']


class PredictionService:

    @staticmethod
    async def predict(features: Dict[str, float], image_filename: str) -> PredictionResponse:
        try:
            model = ModelLoader.get_model()
            scaler = ModelLoader.get_scaler()

            extractor = FeatureExtractor()
            feature_vector = extractor.features_to_vector(features)
            feature_array = np.array([feature_vector], dtype=np.float64)
            feature_scaled = scaler.transform(feature_array)

            prediction = model.predict(feature_scaled)[0]
            probabilities = model.predict_proba(feature_scaled)[0]

            # Distribución con las 4 categorías (inicialmente en 0)
            dist = {c: 0.0 for c in CATEGORIES}
            for i, cls in enumerate(model.classes_):
                if cls in dist:
                    dist[cls] = float(probabilities[i] * 100)

            confidence = float(max(probabilities))

            metadata = ModelLoader.get_metadata()
            model_info = ModelInfo(
                name="vegetation_classifier",
                version=metadata.get("version", "unknown")
            )

            prediction_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)

            # Persistir en DB
            prediction_data = PredictionCreate(
                prediction_id=prediction_id,
                created_at=now,
                model_version=model_info.version,
                predicted_class=str(prediction),
                confidence=confidence,
                semi_humeda_percentage=dist["semi_humeda"],
                seca_percentage=dist["seca"],
                muy_seca_percentage=dist["muy_seca"],
                muy_seca_extrema_percentage=dist["muy_seca_extrema"],
                image_filename=image_filename,
                processing_time_ms=0,
                # 11 features
                green_pixel_ratio=features.get("green_pixel_ratio", 0.0),
                brown_pixel_ratio=features.get("brown_pixel_ratio", 0.0),
                ndvi_sim=features.get("ndvi_simulated", 0.0),
                gci=features.get("gci", 0.0),
                hsv_mean_h=features.get("hsv_mean_h", 0.0),
                hsv_mean_s=features.get("hsv_mean_s", 0.0),
                hsv_mean_v=features.get("hsv_mean_v", 0.0),
                green_brown_ratio=features.get("green_brown_ratio", 0.0),
                green_minus_brown=features.get("green_minus_brown", 0.0),
                ndvi_gci_interaction=features.get("ndvi_gci_interaction", 0.0),
                hue_distance_to_green=features.get(
                    "hue_distance_to_green", 0.0)
            )

            try:
                await PredictionRepository.create(prediction_data)
                logger.info(f"✅ Predicción guardada: {prediction_id}")
            except Exception as e:
                logger.error(f"❌ Error guardando predicción: {e}")

            # ---------- Construcción de respuesta con tipado correcto ----------
            return PredictionResponse(
                prediction_id=prediction_id,
                timestamp=now,
                model=model_info,
                prediction=PredictionDetail(
                    category=str(prediction),
                    confidence=confidence
                ),
                distribution=DistributionDetail(
                    semi_humeda=dist["semi_humeda"],
                    seca=dist["seca"],
                    muy_seca=dist["muy_seca"],
                    muy_seca_extrema=dist["muy_seca_extrema"]
                ),
                features=FeaturesDetail(
                    green_pixel_ratio=float(
                        features.get("green_pixel_ratio", 0.0)),
                    brown_pixel_ratio=float(
                        features.get("brown_pixel_ratio", 0.0)),
                    ndvi_simulated=float(features.get("ndvi_simulated", 0.0)),
                    gci=float(features.get("gci", 0.0)),
                    hsv_mean_h=float(features.get("hsv_mean_h", 0.0)),
                    hsv_mean_s=float(features.get("hsv_mean_s", 0.0)),
                    hsv_mean_v=float(features.get("hsv_mean_v", 0.0)),
                    green_brown_ratio=float(
                        features.get("green_brown_ratio", 0.0)),
                    green_minus_brown=float(
                        features.get("green_minus_brown", 0.0)),
                    ndvi_gci_interaction=float(
                        features.get("ndvi_gci_interaction", 0.0)),
                    hue_distance_to_green=float(
                        features.get("hue_distance_to_green", 0.0))
                )
            )

        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise
