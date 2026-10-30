# app/services/model_service.py
from typing import Dict, Any
from app.ml.model_loader import ModelLoader


class ModelService:

    @staticmethod
    def get_model_info() -> Dict[str, Any]:
        metadata = ModelLoader.get_metadata()
        return {
            "name": "vegetation_classifier",
            "version": metadata.get("version", "unknown"),
            "training_date": metadata.get("training_date", "unknown"),
            "accuracy": float(metadata.get("accuracy", 0.0)),
            "f1_macro_cv_mean": float(metadata.get("f1_macro_cv_mean", 0.0)),
            "f1_macro_cv_std": float(metadata.get("f1_macro_cv_std", 0.0)),
            "n_samples": int(metadata.get("n_samples", 0)),
            "features": ModelLoader.get_feature_names(),
            "classes": list(ModelLoader.get_model().classes_)
        }

    @staticmethod
    def is_available() -> bool:
        return ModelLoader.is_loaded()
