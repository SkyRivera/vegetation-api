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
    PredictionCreate, PredictionResponse, PredictionDetail,
    DistributionDetail, FeaturesDetail, ModelInfo
)

logger = logging.getLogger(__name__)
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
            
            # Scikit-learn ahora devuelve strings ('semi_humeda', etc.)
            prediction = str(model.predict(feature_scaled)[0])
            probabilities = model.predict_proba(feature_scaled)[0]
            
            dist = {c: 0.0 for c in CATEGORIES}
            for i, cls in enumerate(model.classes_):
                if str(cls) in dist:
                    dist[str(cls)] = float(probabilities[i] * 100)
            
            confidence = float(max(probabilities))
            metadata = ModelLoader.get_metadata()
            model_info = ModelInfo(name="vegetation_classifier", version=metadata.get("version", "unknown"))
            
            prediction_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            
            # 1. Preparar datos para la Base de Datos (PredictionCreate)
            prediction_data = PredictionCreate(
                prediction_id=prediction_id,
                created_at=now,
                model_version=model_info.version,
                predicted_class=prediction,
                confidence=confidence,
                semi_humeda_percentage=dist["semi_humeda"],
                seca_percentage=dist["seca"],
                muy_seca_percentage=dist["muy_seca"],
                muy_seca_extrema_percentage=dist["muy_seca_extrema"],
                image_filename=image_filename,
                processing_time_ms=0,
                # Mapeo de las 20 nuevas features
                pct_verde=features.get("pct_verde", 0.0),
                pct_cafe=features.get("pct_cafe", 0.0),
                pct_amarillo=features.get("pct_amarillo", 0.0),
                pct_verde_n=features.get("pct_verde_n", 0.0),
                pct_cafe_n=features.get("pct_cafe_n", 0.0),
                pct_amarillo_n=features.get("pct_amarillo_n", 0.0),
                sat_verde=features.get("sat_verde", 0.0),
                sat_cafe=features.get("sat_cafe", 0.0),
                sat_amarillo=features.get("sat_amarillo", 0.0),
                mean_h_veg=features.get("mean_h_veg", 0.0),
                std_h_veg=features.get("std_h_veg", 0.0),
                lap_var=features.get("lap_var", 0.0),
                lap_var_veg=features.get("lap_var_veg", 0.0),
                lap_abs_mean_veg=features.get("lap_abs_mean_veg", 0.0),
                grad_mean=features.get("grad_mean", 0.0),
                grad_std=features.get("grad_std", 0.0),
                local_var_mean=features.get("local_var_mean", 0.0),
                lbp_entropy=features.get("lbp_entropy", 0.0),
                lbp_std=features.get("lbp_std", 0.0),
                densidad_verde=features.get("densidad_verde", 0.0)
            )
            
            # Guardar en DB
            try:
                await PredictionRepository.create(prediction_data)
                logger.info(f"✅ Predicción guardada: {prediction_id}")
            except Exception as e:
                logger.error(f"❌ Error guardando predicción: {e}")
            
            # 2. Preparar datos para la Respuesta (PredictionResponse)
            return PredictionResponse(
                prediction_id=prediction_id,
                timestamp=now,
                model=model_info,
                prediction=PredictionDetail(category=prediction, confidence=confidence),
                distribution=DistributionDetail(
                    semi_humeda=dist["semi_humeda"],
                    seca=dist["seca"],
                    muy_seca=dist["muy_seca"],
                    muy_seca_extrema=dist["muy_seca_extrema"]
                ),
                features=FeaturesDetail(
                    pct_verde=features.get("pct_verde", 0.0),
                    pct_cafe=features.get("pct_cafe", 0.0),
                    pct_amarillo=features.get("pct_amarillo", 0.0),
                    pct_verde_n=features.get("pct_verde_n", 0.0),
                    pct_cafe_n=features.get("pct_cafe_n", 0.0),
                    pct_amarillo_n=features.get("pct_amarillo_n", 0.0),
                    sat_verde=features.get("sat_verde", 0.0),
                    sat_cafe=features.get("sat_cafe", 0.0),
                    sat_amarillo=features.get("sat_amarillo", 0.0),
                    mean_h_veg=features.get("mean_h_veg", 0.0),
                    std_h_veg=features.get("std_h_veg", 0.0),
                    lap_var=features.get("lap_var", 0.0),
                    lap_var_veg=features.get("lap_var_veg", 0.0),
                    lap_abs_mean_veg=features.get("lap_abs_mean_veg", 0.0),
                    grad_mean=features.get("grad_mean", 0.0),
                    grad_std=features.get("grad_std", 0.0),
                    local_var_mean=features.get("local_var_mean", 0.0),
                    lbp_entropy=features.get("lbp_entropy", 0.0),
                    lbp_std=features.get("lbp_std", 0.0),
                    densidad_verde=features.get("densidad_verde", 0.0)
                )
            )
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise