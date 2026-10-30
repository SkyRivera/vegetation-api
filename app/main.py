# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.api.routes import health, model, predictions
from app.ml.model_loader import ModelLoader
from app.db.database import init_db

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando aplicación...")

    try:
        ModelLoader.load_model(settings.MODEL_PATH)
        logger.info(f"✅ Modelo cargado desde {settings.MODEL_PATH}")
        logger.info(f"   Features ({len(ModelLoader.get_feature_names())}): "
                    f"{ModelLoader.get_feature_names()}")
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")
        raise

    try:
        await init_db()
        logger.info("✅ Base de datos inicializada")
    except Exception as e:
        logger.error(f"❌ Error inicializando DB: {e}")
        raise

    yield
    logger.info("Cerrando aplicación...")


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(model.router, prefix="/api/v1")
app.include_router(predictions.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs": "/docs",
        "status": "running",
        "categories": ["semi_humeda", "seca", "muy_seca", "muy_seca_extrema"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS
    )
