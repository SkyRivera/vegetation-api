# app/api/routes/health.py
from fastapi import APIRouter, status
from app.ml.model_loader import ModelLoader

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": ModelLoader.is_loaded(),
        "service": "vegetation_classification_api",
        "categories": ["semi_humeda", "seca", "muy_seca", "muy_seca_extrema"],
        "n_features": len(ModelLoader.get_feature_names())
    }
