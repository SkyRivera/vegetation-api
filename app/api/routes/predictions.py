# app/api/routes/predictions.py
from fastapi import APIRouter, HTTPException, UploadFile, File, status
import time

from app.services.image_service import ImageService
from app.services.prediction_service import PredictionService
from app.ml.model_loader import ModelLoader
from app.schemas.prediction import PredictionResponse

router = APIRouter()
image_service = ImageService()


@router.post("/vegetation/predict", response_model=PredictionResponse)
async def predict_vegetation(
    image: UploadFile = File(..., description="Imagen de vegetación")
):
    """
    Clasifica la vegetación de la imagen en 4 categorías:
    `semi_humeda`, `seca`, `muy_seca`, `muy_seca_extrema`.
    """
    if not ModelLoader.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )

    try:
        image_service.validate_image(image)

        start_time = time.time()
        features = image_service.process_image(image)
        processing_time = int((time.time() - start_time) * 1000)

        prediction = await PredictionService.predict(
            features=features,
            image_filename=image.filename or "unknown"
        )

        return prediction

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando imagen: {str(e)}"
        )
