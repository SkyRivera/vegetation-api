# app/utils/validators.py
import imghdr
from fastapi import UploadFile
from app.config import settings


def validate_image_file(file: UploadFile) -> None:
    filename = file.filename.lower()
    allowed_extensions = [f".{fmt}" for fmt in settings.ALLOWED_IMAGE_FORMATS]

    if not any(filename.endswith(ext) for ext in allowed_extensions):
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
