================================================================================
VEGETATION CLASSIFICATION API
Clasificación de Vegetación mediante Visión Artificial
================================================================================

Versión: 1.1.0
Estado: Estable
Licencia: MIT (ver archivo LICENSE)
Documentación: DOC-API-CON-001
Repositorio: [Público]
Clasificación: Público

================================================================================

1. # DESCRIPCIÓN

La Vegetation Classification API es un servicio REST de producción que
clasifica el estado de la vegetación a partir de imágenes (JPG, JPEG, PNG)
utilizando técnicas de Computer Vision y un modelo de Machine Learning
supervisado (RandomForestClassifier).

El sistema determina una de las CUATRO categorías de vegetación, alineadas
con el concepto de semáforo ambiental:

┌─────────────────────┬──────────────┬────────────────────────────────────┐
│ Categoría │ Semáforo │ Descripción │
├─────────────────────┼──────────────┼────────────────────────────────────┤
│ semi_humeda │ Verde │ Vegetación con cobertura verde │
│ seca │ Amarillo │ Vegetación con sequedad incipiente │
│ muy_seca │ Rojo │ Predominio de vegetación seca │
│ muy_seca_extrema │ Rojo osc. │ Suelo árido / vegetación muerta │
└─────────────────────┴──────────────┴────────────────────────────────────┘

Características principales:

• Procesamiento de imágenes con OpenCV.
• Extracción de 11 características (7 base + 4 derivadas).
• Inferencia con RandomForestClassifier (scikit-learn).
• Persistencia de cada predicción en PostgreSQL.
• API REST documentada con OpenAPI (Swagger UI + ReDoc).
• Arquitectura en capas simple y mantenible.
• Despliegue en contenedores Docker sobre un único servidor Linux.
• Proxy inverso con Nginx.

================================================================================ 2. ARQUITECTURA
================================================================================

Flujo de una petición:

Cliente
│
│ HTTPS / multipart/form-data + JSON
▼
Nginx (Reverse Proxy, :80)
│
▼
FastAPI + Uvicorn (:8000)
│
├── Routes → Endpoints HTTP
├── Schemas → Contratos Pydantic
├── Services → Lógica de negocio
├── ML → FeatureExtractor + ModelLoader
└── DB → Repositorios (SQLAlchemy async)
│
├── PostgreSQL → Persistencia de predicciones
└── Filesystem → Modelo .joblib

Estructura de directorios del proyecto:

vegetation-api/
│
├── app/
│ ├── main.py # Punto de entrada FastAPI
│ ├── config.py # Configuración (Pydantic Settings)
│ ├── api/routes/
│ │ ├── health.py # GET /health
│ │ ├── model.py # GET /api/v1/model
│ │ └── predictions.py # POST /api/v1/vegetation/predict
│ ├── schemas/
│ │ ├── prediction.py # Modelos Pydantic de predicción
│ │ └── model.py # Modelos Pydantic del modelo
│ ├── services/
│ │ ├── prediction_service.py # Orquestación de predicción
│ │ ├── image_service.py # Validación y procesamiento
│ │ └── model_service.py # Gestión del modelo
│ ├── ml/
│ │ ├── feature_extractor.py # Extracción de 11 features
│ │ └── model_loader.py # Carga única del modelo
│ ├── db/
│ │ ├── database.py # Conexión PostgreSQL
│ │ ├── models.py # ORM SQLAlchemy
│ │ └── repositories.py # CRUD
│ └── utils/
│ ├── logger.py
│ └── validators.py
│
├── training/
│ └── train.py # Script de entrenamiento offline
│
├── models/
│ └── vegetation_classifier_latest.joblib
│
├── tests/
│ ├── test_health.py
│ ├── test_predictions.py
│ └── test_model.py
│
├── docker/
│ ├── Dockerfile.fastapi
│ └── Dockerfile.nginx
│
├── nginx/
│ └── nginx.conf
│
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
├── README.txt
└── LICENSE

================================================================================ 3. STACK TECNOLÓGICO
================================================================================

Lenguaje: Python 3.10
Framework Web: FastAPI 0.104.1
Servidor ASGI: Uvicorn 0.24.0
Validación: Pydantic 2.5.0 + pydantic-settings 2.1.0
Computer Vision: OpenCV 4.8.1 (headless) + NumPy 1.24.3
Machine Learning: scikit-learn 1.3.2 + joblib 1.3.2
Base de datos: PostgreSQL 15 + SQLAlchemy 2.0.23 (async)
Driver DB: asyncpg 0.29.0
Proxy inverso: Nginx (Alpine)
Contenedores: Docker + Docker Compose
Pruebas: pytest 7.4.3 + httpx 0.25.1

================================================================================ 4. REQUISITOS DEL SISTEMA
================================================================================

Hardware (mínimo recomendado para un único servidor):

• CPU: 2 vCPU
• RAM: 4 GB (para 4 workers × modelo en memoria)
• Disco: 20 GB
• Red: Puerto 80 (HTTP) o 443 (HTTPS) accesible por clientes

Software:

• Docker >= 20.10
• Docker Compose >= 2.0
• Git (opcional, para despliegue desde repositorio)
• (Desarrollo) Python 3.12.2

================================================================================ 5. INSTALACIÓN Y DESPLIEGUE (DOCKER)
================================================================================

## 5.1. Preparación del servidor Linux

sudo apt update && sudo apt upgrade -y
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER # requiere re-login
docker --version
docker compose version

## 5.2. Obtención del código

sudo mkdir -p /opt/vegetation-api
cd /opt/vegetation-api
git clone <URL_DEL_REPOSITORIO> .

# o copiar manualmente el árbol del proyecto

## 5.3. Configuración del entorno

cp .env.example .env
nano .env # ajustar credenciales y parámetros

Variables obligatorias:

DB_HOST=postgres
DB_PORT=5432
DB_NAME=vegetation_db
DB_USER=vegetation_user
DB_PASSWORD=<CAMBIAR_EN_PRODUCCION>
MODEL_PATH=models/vegetation_classifier_latest.joblib
LOG_LEVEL=INFO
WORKERS=4

## 5.4. Colocación del modelo entrenado

mkdir -p models
cp /ruta/al/vegetation_classifier_v\*.joblib models/
ln -sf vegetation_classifier_v20260918_120000.joblib \
 models/vegetation_classifier_latest.joblib

## 5.5. Construcción y arranque

docker compose build
docker compose up -d
docker compose ps

## 5.6. Verificación

curl http://localhost/health
curl http://localhost/api/v1/model
curl -X POST -F "image=@test.jpg" \
 http://localhost/api/v1/vegetation/predict

## 5.7. Documentación interactiva

• Swagger UI: http://localhost/docs
• ReDoc: http://localhost/redoc
• OpenAPI JSON: http://localhost/openapi.json

================================================================================ 6. INSTALACIÓN Y DESPLIEGUE (LOCAL / DESARROLLO)
================================================================================

python3.10 -m venv venv
source venv/bin/activate # Linux/macOS

# .\venv\Scripts\activate # Windows

pip install -r requirements.txt

# Crear modelo si no existe

python training/train.py

# Levantar FastAPI

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

================================================================================ 7. ENDPOINTS DE LA API
================================================================================

Base URL: http://{host}:{port}

┌────────┬──────────────────────────────────┬─────────────────────────────────┐
│ Método │ Endpoint │ Descripción │
├────────┼──────────────────────────────────┼─────────────────────────────────┤
│ GET │ / │ Info del servicio │
│ GET │ /health │ Health check │
│ GET │ /api/v1/model │ Metadata del modelo ML │
│ POST │ /api/v1/vegetation/predict │ Clasificación de vegetación │
└────────┴──────────────────────────────────┴─────────────────────────────────┘

## 7.1. GET /health

Respuesta 200:

{
"status": "healthy",
"model_loaded": true,
"service": "vegetation_classification_api",
"categories": ["semi_humeda", "seca", "muy_seca", "muy_seca_extrema"],
"n_features": 11
}

## 7.2. GET /api/v1/model

Respuesta 200:

{
"name": "vegetation_classifier",
"version": "v20260918_120000",
"training_date": "2026-09-18T12:00:00",
"accuracy": 0.96,
"f1_macro_cv_mean": 0.94,
"f1_macro_cv_std": 0.02,
"n_samples": 4000,
"features": ["green_pixel_ratio", "brown_pixel_ratio",
"ndvi_simulated", "gci", "hsv_mean_h",
"hsv_mean_s", "hsv_mean_v",
"green_brown_ratio", "green_minus_brown",
"ndvi_gci_interaction", "hue_distance_to_green"],
"classes": ["semi_humeda", "seca", "muy_seca", "muy_seca_extrema"]
}

Respuesta 503 si el modelo no está cargado.

## 7.3. POST /api/v1/vegetation/predict

Request:

Content-Type: multipart/form-data
Campo: image (archivo JPG/JPEG/PNG ≤ 10 MB)

Respuesta 200:

{
"prediction_id": "123e4567-e89b-12d3-a456-426614174000",
"timestamp": "2026-09-18T22:00:00Z",
"model": {
"name": "vegetation_classifier",
"version": "v20260918_120000"
},
"prediction": {
"category": "seca",
"confidence": 0.87
},
"distribution": {
"semi_humeda": 12.2,
"seca": 66.3,
"muy_seca": 16.4,
"muy_seca_extrema": 5.1
},
"features": {
"green_pixel_ratio": 0.30,
"brown_pixel_ratio": 0.45,
"ndvi_simulated": 0.10,
"gci": 0.05,
"hsv_mean_h": 40.2,
"hsv_mean_s": 110.4,
"hsv_mean_v": 130.1,
"green_brown_ratio": 0.667,
"green_minus_brown": -0.15,
"ndvi_gci_interaction": 0.005,
"hue_distance_to_green": 0.33
}
}

Errores:

• 400 Bad Request → formato/tamaño/corrupción
• 422 Unprocessable → falta el campo image
• 500 Internal Server → error inesperado
• 503 Service Unavailable→ modelo no disponible

================================================================================ 8. CÓDIGOS DE ERROR
================================================================================

┌──────┬───────────────────────────────────────────────────────────────────┐
│ HTTP │ Significado │
├──────┼───────────────────────────────────────────────────────────────────┤
│ 200 │ OK │
│ 400 │ Bad Request — validación de imagen fallida │
│ 422 │ Unprocessable Entity — falta campo obligatorio │
│ 500 │ Internal Server Error — excepción no controlada │
│ 503 │ Service Unavailable — modelo no disponible │
└──────┴───────────────────────────────────────────────────────────────────┘

Formato de error (estándar FastAPI):

{ "detail": "<mensaje>" }

================================================================================ 9. EJEMPLOS DE CONSUMO
================================================================================

## 9.1. cURL

# Health

curl http://localhost:8000/health

# Info del modelo

curl http://localhost:8000/api/v1/model

# Clasificación

curl -X POST -F "image=@vegetacion.jpg" \
 http://localhost:8000/api/v1/vegetation/predict

## 9.2. Python (requests)

import requests

BASE = "http://localhost:8000"

# Health

print(requests.get(f"{BASE}/health").json())

# Model info

print(requests.get(f"{BASE}/api/v1/model").json())

# Predict

with open("vegetacion.jpg", "rb") as f:
files = {"image": ("vegetacion.jpg", f, "image/jpeg")}
r = requests.post(f"{BASE}/api/v1/vegetation/predict", files=files)

print(r.status_code, r.json())

## 9.3. JavaScript (fetch)

const formData = new FormData();
formData.append('image', fileInput.files[0]);

const response = await fetch(
'http://localhost:8000/api/v1/vegetation/predict',
{ method: 'POST', body: formData }
);

const result = await response.json();
console.log(result.prediction.category, result.prediction.confidence);

## 9.4. Postman

Method: POST
URL: http://localhost:8000/api/v1/vegetation/predict
Body: form-data
Key: image
Type: File
Value: <seleccionar imagen>

================================================================================ 10. ENTRENAMIENTO DEL MODELO
================================================================================

El modelo se entrena FUERA de la API, mediante un script independiente:

python training/train.py

Este proceso:

1. Genera datos sintéticos de entrenamiento (4000 muestras).
2. Entrena un RandomForestClassifier (300 árboles, max_depth=15).
3. Calcula métricas (accuracy, F1 macro CV 5-fold).
4. Persiste el modelo + scaler + metadata en un archivo .joblib.
5. Genera versión con timestamp (vYYYYMMDD_HHMMSS).

Salida:

models/vegetation_classifier_vYYYYMMDD_HHMMSS.joblib
models/vegetation_classifier_latest.joblib

IMPORTANTE:
• La API NO entrena modelos.
• El modelo se carga UNA SOLA VEZ al iniciar la aplicación (lifespan).
• El orden de las 11 features debe coincidir entre train.py y
FeatureExtractor.FEATURE_NAMES.

================================================================================ 11. PRUEBAS
================================================================================

Ejecutar la suite de pruebas:

pytest tests/ -v

Cobertura recomendada:

pytest --cov=app tests/

Pruebas incluidas:

• test_health.py → /health responde 200 y model_loaded
• test_model.py → metadata del modelo
• test_predictions.py → validación de tipado y errores
• test_features.py → extracción de 11 features

================================================================================ 12. MANTENIMIENTO
================================================================================

## 12.1. Logs

docker compose logs -f
docker compose logs -f fastapi
docker compose logs -f nginx
docker compose logs -f postgres

## 12.2. Actualización del modelo

# 1. Copiar nuevo modelo al servidor

scp models/vegetation_classifier_vNUEVO.joblib \
 usuario@servidor:/opt/vegetation-api/models/

# 2. Actualizar symlink

cd /opt/vegetation-api/models
ln -sf vegetation_classifier_vNUEVO.joblib \
 vegetation_classifier_latest.joblib

# 3. Reiniciar FastAPI

docker compose restart fastapi

# 4. Verificar

curl http://localhost/api/v1/model

## 12.3. Actualización del código

cd /opt/vegetation-api
git pull
docker compose build fastapi
docker compose up -d fastapi

## 12.4. Backup de PostgreSQL

# Crear backup

docker exec vegetation*postgres pg_dump -U vegetation_user vegetation_db \
 > backup*$(date +%Y%m%d\_%H%M%S).sql

# Restaurar

cat backup_20260922.sql | \
 docker exec -i vegetation_postgres psql -U vegetation_user vegetation_db

## 12.5. Rollback

# Revertir a commit anterior

git checkout <COMMIT_ANTERIOR>
docker compose build fastapi
docker compose up -d fastapi

# O revertir el modelo

cd models
ln -sf vegetation_classifier_vANTERIOR.joblib \
 vegetation_classifier_latest.joblib
docker compose restart fastapi

## 12.6. Monitoreo básico

docker compose ps
docker stats --no-stream
curl http://localhost/health

================================================================================ 13. SEGURIDAD
================================================================================

Estado actual (v1.1.0):

⚠️ No existe autenticación ni autorización.
⚠️ CORS configurado con allow_origins=["*"] (inseguro en producción).
⚠️ HTTPS no configurado por defecto.

Recomendaciones antes de exponer la API públicamente:

1. Configurar HTTPS con Let's Encrypt en Nginx.
2. Implementar API Key o JWT (OAuth 2.0 / Bearer).
3. Restringir CORS a orígenes específicos.
4. Implementar rate limiting (slowapi o Nginx limit_req).
5. Rotar credenciales de PostgreSQL periódicamente.
6. Auditar logs periódicamente.

NO exponga esta API a Internet sin aplicar al menos los puntos 1–4.

================================================================================ 14. LIMITACIONES CONOCIDAS
================================================================================

1. NDVI simulado: usa canal verde como aproximación de NIR.
   NO es NDVI real. Para NDVI real se requiere cámara multiespectral
   con canal NIR.

2. Fisheye: no hay corrección de distorsión real (falta calibración
   de cámara). Se documenta como "simplificada".

3. El modelo fue entrenado con datos sintéticos; se recomienda
   reentrenamiento con datos reales anotados.

4. La API persiste el nombre de archivo de imagen pero NO el binario.

5. No existen endpoints de historial de predicciones expuestos
   (aunque el repositorio existe).

================================================================================ 15. SOPORTE Y CONTACTO
================================================================================

Owner técnico: Equipo de Visión Artificial y Monitoreo Ambiental
Email: veg-api-owner@domain.com
SRE: sre-vegetation@domain.com
Canal de escalación: #sre-vegetation

Documento de contrato API: DOC-API-CON-001
Especificación OpenAPI: /openapi.json

================================================================================ 16. LICENCIA
================================================================================

Este software se distribuye bajo la Licencia MIT.
Consulte el archivo LICENSE para el texto completo.

================================================================================
FIN DEL DOCUMENTO
================================================================================
