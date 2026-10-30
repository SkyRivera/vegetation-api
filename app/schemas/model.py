# app/schemas/model.py
from pydantic import BaseModel
from typing import List


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    training_date: str
    accuracy: float
    f1_macro_cv_mean: float
    f1_macro_cv_std: float
    n_samples: int
    features: List[str]
    classes: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "name": "vegetation_classifier",
                "version": "v20260918_120000",
                "training_date": "2026-09-18T12:00:00",
                "accuracy": 0.96,
                "f1_macro_cv_mean": 0.94,
                "f1_macro_cv_std": 0.02,
                "n_samples": 4000,
                "features": [
                    "green_pixel_ratio", "brown_pixel_ratio",
                    "ndvi_simulated", "gci", "hsv_mean_h",
                    "hsv_mean_s", "hsv_mean_v",
                    "green_brown_ratio", "green_minus_brown",
                    "ndvi_gci_interaction", "hue_distance_to_green"
                ],
                "classes": ["semi_humeda", "seca", "muy_seca", "muy_seca_extrema"]
            }
        }
