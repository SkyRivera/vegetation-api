# app/ml/model_loader.py
import joblib
import os
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Carga y gestiona el modelo ML."""

    _model = None
    _scaler = None
    _feature_names = None
    _metadata = None
    _color_thresholds = None

    @classmethod
    def load_model(cls, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        try:
            model_data = joblib.load(model_path)

            cls._model = model_data['model']
            cls._scaler = model_data['scaler']
            cls._feature_names = model_data.get('feature_names', [])
            cls._metadata = model_data.get('metadata', {})
            cls._color_thresholds = model_data.get('color_thresholds', {})

            logger.info(
                f"✅ Modelo cargado: {cls._metadata.get('version', 'unknown')}")
            logger.info(f"   Clases: {list(cls._model.classes_)}")
            logger.info(f"   Features: {len(cls._feature_names)}")

            # Validación: el scaler debe esperar el mismo nº de features
            n_scaler = getattr(cls._scaler, 'n_features_in_', None)
            if n_scaler is not None and n_scaler != len(cls._feature_names):
                raise ValueError(
                    f"Desajuste: scaler espera {n_scaler} features, "
                    f"pero se declararon {len(cls._feature_names)}"
                )

        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise

    @classmethod
    def get_model(cls):
        if cls._model is None:
            raise ValueError("El modelo no ha sido cargado")
        return cls._model

    @classmethod
    def get_scaler(cls):
        if cls._scaler is None:
            raise ValueError("El scaler no ha sido cargado")
        return cls._scaler

    @classmethod
    def get_feature_names(cls) -> List[str]:
        return cls._feature_names or []

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        return cls._metadata or {}

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model is not None
