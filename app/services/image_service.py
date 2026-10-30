# app/services/image_service.py
from typing import Dict
from fastapi import UploadFile
import logging

from app.ml.feature_extractor import FeatureExtractor
from app.config import settings

logger = logging.getLogger(__name__)


class ImageService:

    def __init__(self):
        self.feature_extractor = FeatureExtractor(
            target_size=settings.IMAGE_TARGET_SIZE
        )

    def validate_image(self, file: UploadFile) -> None:
        filename = (file.filename or "").lower()
        allowed = [f".{fmt}" for fmt in settings.ALLOWED_IMAGE_FORMATS]

        if not any(filename.endswith(fmt) for fmt in allowed):
            raise ValueError(
                f"Formato no soportado. Permitidos: {', '.join(settings.ALLOWED_IMAGE_FORMATS)}"
            )

        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)

        max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
        if size > max_bytes:
            raise ValueError(
                f"Imagen demasiado grande. Máximo: {settings.MAX_IMAGE_SIZE_MB}MB"
            )

        if size == 0:
            raise ValueError("El archivo está vacío")

    def process_image(self, file: UploadFile) -> Dict[str, float]:
        try:
            contents = file.file.read()
            image = self.feature_extractor.preprocess_image(contents)
            features = self.feature_extractor.extract_features(image)
            return features
        except Exception as e:
            logger.error(f"Error procesando imagen: {e}")
            raise
