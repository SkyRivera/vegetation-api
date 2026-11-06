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
    created_at = Column(DateTime(timezone=True), nullable=False) # <-- Fix de fecha aplicado aquí
    model_version = Column(String(50), nullable=False)
    
    predicted_class = Column(String(30), nullable=False)
    confidence = Column(Float, nullable=False)
    semi_humeda_percentage = Column(Float, nullable=False)
    seca_percentage = Column(Float, nullable=False)
    muy_seca_percentage = Column(Float, nullable=False)
    muy_seca_extrema_percentage = Column(Float, nullable=False)
    
    image_filename = Column(String(255), nullable=False)
    processing_time_ms = Column(Integer, nullable=False)
    
    # 11 Cromáticas
    pct_verde = Column(Float, nullable=False)
    pct_cafe = Column(Float, nullable=False)
    pct_amarillo = Column(Float, nullable=False)
    pct_verde_n = Column(Float, nullable=False)
    pct_cafe_n = Column(Float, nullable=False)
    pct_amarillo_n = Column(Float, nullable=False)
    sat_verde = Column(Float, nullable=False)
    sat_cafe = Column(Float, nullable=False)
    sat_amarillo = Column(Float, nullable=False)
    mean_h_veg = Column(Float, nullable=False)
    std_h_veg = Column(Float, nullable=False)
    
    # 9 Textura
    lap_var = Column(Float, nullable=False)
    lap_var_veg = Column(Float, nullable=False)
    lap_abs_mean_veg = Column(Float, nullable=False)
    grad_mean = Column(Float, nullable=False)
    grad_std = Column(Float, nullable=False)
    local_var_mean = Column(Float, nullable=False)
    lbp_entropy = Column(Float, nullable=False)
    lbp_std = Column(Float, nullable=False)
    densidad_verde = Column(Float, nullable=False)

    __table_args__ = (
        Index('ix_predictions_created_at', created_at),
        Index('ix_predictions_predicted_class', predicted_class),
    )