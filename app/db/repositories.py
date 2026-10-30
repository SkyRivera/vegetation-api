# app/db/repositories.py
from sqlalchemy import select, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from typing import Optional, List, Tuple

from app.db.database import AsyncSessionLocal
from app.db.models import Prediction
from app.schemas.prediction import PredictionCreate


class PredictionRepository:

    @staticmethod
    async def create(prediction_data: PredictionCreate) -> Prediction:
        async with AsyncSessionLocal() as session:
            prediction = Prediction(
                prediction_id=prediction_data.prediction_id,
                created_at=prediction_data.created_at,
                model_version=prediction_data.model_version,
                predicted_class=prediction_data.predicted_class,
                confidence=prediction_data.confidence,
                semi_humeda_percentage=prediction_data.semi_humeda_percentage,
                seca_percentage=prediction_data.seca_percentage,
                muy_seca_percentage=prediction_data.muy_seca_percentage,
                muy_seca_extrema_percentage=prediction_data.muy_seca_extrema_percentage,
                image_filename=prediction_data.image_filename,
                processing_time_ms=prediction_data.processing_time_ms,
                # 7 base
                green_pixel_ratio=prediction_data.green_pixel_ratio,
                brown_pixel_ratio=prediction_data.brown_pixel_ratio,
                ndvi_sim=prediction_data.ndvi_sim,
                gci=prediction_data.gci,
                hsv_mean_h=prediction_data.hsv_mean_h,
                hsv_mean_s=prediction_data.hsv_mean_s,
                hsv_mean_v=prediction_data.hsv_mean_v,
                # 4 derivadas
                green_brown_ratio=prediction_data.green_brown_ratio,
                green_minus_brown=prediction_data.green_minus_brown,
                ndvi_gci_interaction=prediction_data.ndvi_gci_interaction,
                hue_distance_to_green=prediction_data.hue_distance_to_green
            )

            session.add(prediction)
            await session.commit()
            await session.refresh(prediction)
            return prediction

    @staticmethod
    async def get_by_id(prediction_id: str) -> Optional[Prediction]:
        async with AsyncSessionLocal() as session:
            query = select(Prediction).where(
                Prediction.prediction_id == prediction_id
            )
            result = await session.execute(query)
            return result.scalar_one_or_none()

    @staticmethod
    async def get_list(
        page: int = 1,
        limit: int = 20,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        category: Optional[str] = None
    ) -> Tuple[List[Prediction], int]:
        async with AsyncSessionLocal() as session:
            query = select(Prediction)
            count_query = select(Prediction)

            filters = []
            if date_from:
                filters.append(Prediction.created_at >= date_from)
            if date_to:
                filters.append(Prediction.created_at <= date_to)
            if category:
                filters.append(Prediction.predicted_class == category)

            if filters:
                query = query.where(and_(*filters))
                count_query = count_query.where(and_(*filters))

            offset = (page - 1) * limit
            query = query.order_by(desc(Prediction.created_at)).offset(
                offset).limit(limit)

            total_result = await session.execute(count_query)
            total = len(total_result.scalars().all())

            result = await session.execute(query)
            items = result.scalars().all()
            return items, total
