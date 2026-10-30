# app/db/models.py
from sqlalchemy import Column, String, Float, Integer, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

from app.db.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(String(36), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    model_version = Column(String(50), nullable=False)

    # Resultado
    predicted_class = Column(String(30), nullable=False)
    confidence = Column(Float, nullable=False)
    semi_humeda_percentage = Column(Float, nullable=False)
    seca_percentage = Column(Float, nullable=False)
    muy_seca_percentage = Column(Float, nullable=False)
    muy_seca_extrema_percentage = Column(Float, nullable=False)

    # Metadata
    image_filename = Column(String(255), nullable=False)
    processing_time_ms = Column(Integer, nullable=False)

    # Features base (7)
    green_pixel_ratio = Column(Float, nullable=False)
    brown_pixel_ratio = Column(Float, nullable=False)
    ndvi_sim = Column(Float, nullable=False)
    gci = Column(Float, nullable=False)
    hsv_mean_h = Column(Float, nullable=False)
    hsv_mean_s = Column(Float, nullable=False)
    hsv_mean_v = Column(Float, nullable=False)

    # Features derivadas (4) - NUEVAS
    green_brown_ratio = Column(Float, nullable=False, default=0.0)
    green_minus_brown = Column(Float, nullable=False, default=0.0)
    ndvi_gci_interaction = Column(Float, nullable=False, default=0.0)
    hue_distance_to_green = Column(Float, nullable=False, default=0.0)

    __table_args__ = (
        Index('ix_predictions_created_at', created_at),
        Index('ix_predictions_predicted_class', predicted_class),
    )
