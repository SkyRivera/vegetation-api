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
                # 11 Cromáticas
                pct_verde=prediction_data.pct_verde,
                pct_cafe=prediction_data.pct_cafe,
                pct_amarillo=prediction_data.pct_amarillo,
                pct_verde_n=prediction_data.pct_verde_n,
                pct_cafe_n=prediction_data.pct_cafe_n,
                pct_amarillo_n=prediction_data.pct_amarillo_n,
                sat_verde=prediction_data.sat_verde,
                sat_cafe=prediction_data.sat_cafe,
                sat_amarillo=prediction_data.sat_amarillo,
                mean_h_veg=prediction_data.mean_h_veg,
                std_h_veg=prediction_data.std_h_veg,
                # 9 Textura
                lap_var=prediction_data.lap_var,
                lap_var_veg=prediction_data.lap_var_veg,
                lap_abs_mean_veg=prediction_data.lap_abs_mean_veg,
                grad_mean=prediction_data.grad_mean,
                grad_std=prediction_data.grad_std,
                local_var_mean=prediction_data.local_var_mean,
                lbp_entropy=prediction_data.lbp_entropy,
                lbp_std=prediction_data.lbp_std,
                densidad_verde=prediction_data.densidad_verde
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
