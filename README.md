# Vegetation Classification API

API REST desarrollada en FastAPI que clasifica el estado de la vegetación a partir de imágenes utilizando Computer Vision (OpenCV) y un modelo supervisado de Machine Learning (`RandomForestClassifier`).

## 🏷️ Categorías de Clasificación

El modelo clasifica la vegetación en 4 estados:

| Categoría          | Semáforo    | Descripción                                    |
| :----------------- | :---------- | :--------------------------------------------- |
| `semi_humeda`      | Verde       | Vegetación con predominio de cobertura verde.  |
| `seca`             | Amarillo    | Vegetación con características de sequedad.    |
| `muy_seca`         | Rojo        | Predominio de vegetación seca o muerta.        |
| `muy_seca_extrema` | Rojo oscuro | Suelo árido, ramas secas o vegetación ausente. |

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.12.2
- **Framework Web:** FastAPI + Uvicorn
- **Machine Learning:** Scikit-learn, Joblib, NumPy
- **Visión Artificial:** OpenCV (opencv-python-headless)
- **Base de Datos:** PostgreSQL + SQLAlchemy (async) + asyncpg
- **Infraestructura:** Docker, Docker Compose, Nginx

## 📂 Estructura del Proyecto

```text
vegetation-api/
├── app/                  # Código fuente de la API (rutas, servicios, ML, DB)
├── training/             # Script de entrenamiento offline (train.py)
├── models/               # Almacenamiento del modelo (.joblib)
├── tests/                # Pruebas unitarias (pytest)
├── docker/               # Dockerfiles (FastAPI y Nginx)
├── nginx/                # Configuración del proxy inverso
├── docker-compose.yml    # Orquestación de servicios
├── requirements.txt      # Dependencias de Python
└── .env.example          # Plantilla de variables de entorno
```
