# app/api/routes/model.py
from fastapi import APIRouter, HTTPException, status
from app.services.model_service import ModelService
from app.schemas.model import ModelInfoResponse

router = APIRouter()


@router.get("/model", response_model=ModelInfoResponse)
async def get_model_info():
    if not ModelService.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )
    return ModelService.get_model_info()
