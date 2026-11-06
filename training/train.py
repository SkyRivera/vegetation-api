# training/train.py
import os
import sys
import glob
import warnings
import numpy as np
import cv2
import joblib
import logging

from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 0. CONFIGURACIÓN GLOBAL
# =============================================================================
IMG_SIZE = (512, 512)
CLASS_NAMES = {
    "semi_humeda": "Semi-húmeda",
    "seca": "Seca",
    "muy_seca": "Muy seca",
    "muy_seca_extrema": "Muy seca extrema",
}
CLASS_FOLDERS = {
    "semi_humeda": "0",       # Carpeta 0 -> etiqueta "semi_humeda"
    "seca": "1",              # Carpeta 1 -> etiqueta "seca"
    "muy_seca": "2",          # Carpeta 2 -> etiqueta "muy_seca"
    "muy_seca_extrema": "3",  # Carpeta 3 -> etiqueta "muy_seca_extrema"
}

DATASET_DIR = "data"
SAVE_DIR = "models"
MODEL_FILENAME = "vegetation_classifier_latest.joblib"
RANDOM_STATE = 42

# Nombres exactos de las 20 características
FEATURE_NAMES = [
    'pct_verde', 'pct_cafe', 'pct_amarillo',
    'pct_verde_n', 'pct_cafe_n', 'pct_amarillo_n',
    'sat_verde', 'sat_cafe', 'sat_amarillo',
    'mean_h_veg', 'std_h_veg',
    'lap_var', 'lap_var_veg', 'lap_abs_mean_veg',
    'grad_mean', 'grad_std',
    'local_var_mean', 'lbp_entropy', 'lbp_std',
    'densidad_verde'
]


# =============================================================================
# 1. EXTRACCIÓN DE CARACTERÍSTICAS
# =============================================================================
def _build_vegetation_mask(hsv: np.ndarray) -> np.ndarray:
    h, s, v = cv2.split(hsv)
    mask_hue = cv2.inRange(h, 10, 95)
    mask_sat = cv2.inRange(s, 45, 255)
    mask_val = cv2.inRange(v, 30, 255)
    mask = cv2.bitwise_and(mask_hue, mask_sat)
    mask = cv2.bitwise_and(mask, mask_val)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _chromatic_signatures(hsv: np.ndarray, veg_mask: np.ndarray) -> list:
    h, s, v = cv2.split(hsv)
    verde = cv2.inRange(hsv, (30, 60, 40), (90, 255, 255))
    cafe1 = cv2.inRange(hsv, (5, 70, 30), (20, 255, 190))
    cafe2 = cv2.inRange(hsv, (0, 60, 30), (5, 255, 170))
    cafe = cv2.bitwise_or(cafe1, cafe2)
    amarillo = cv2.inRange(hsv, (20, 50, 100), (32, 255, 255))

    verde = cv2.bitwise_and(verde, veg_mask)
    cafe = cv2.bitwise_and(cafe, veg_mask)
    amarillo = cv2.bitwise_and(amarillo, veg_mask)

    total_veg = int(np.count_nonzero(veg_mask)) + 1e-6
    pct_verde = np.count_nonzero(verde) / total_veg
    pct_cafe = np.count_nonzero(cafe) / total_veg
    pct_amarillo = np.count_nonzero(amarillo) / total_veg

    suma = pct_verde + pct_cafe + pct_amarillo + 1e-6
    pct_verde_n = pct_verde / suma
    pct_cafe_n = pct_cafe / suma
    pct_amarillo_n = pct_amarillo / suma

    def mean_sat(mask):
        if np.count_nonzero(mask) == 0:
            return 0.0
        return float(s[mask > 0].mean()) / 255.0

    sat_verde = mean_sat(verde)
    sat_cafe = mean_sat(cafe)
    sat_amarillo = mean_sat(amarillo)

    h_veg_vals = h[veg_mask > 0]
    if h_veg_vals.size == 0:
        mean_h, std_h = 0.0, 0.0
    else:
        mean_h = float(h_veg_vals.mean()) / 180.0
        std_h = float(h_veg_vals.std()) / 180.0

    return [
        pct_verde, pct_cafe, pct_amarillo,
        pct_verde_n, pct_cafe_n, pct_amarillo_n,
        sat_verde, sat_cafe, sat_amarillo,
        mean_h, std_h,
    ]


def _texture_features(gray: np.ndarray, veg_mask: np.ndarray) -> list:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    lap_veg = lap[veg_mask > 0]
    lap_var_veg = float(lap_veg.var()) if lap_veg.size > 10 else 0.0
    lap_abs_mean_veg = float(np.abs(lap_veg).mean()) if lap_veg.size > 10 else 0.0

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag_veg = mag[veg_mask > 0]
    grad_mean = float(mag_veg.mean()) if mag_veg.size > 10 else 0.0
    grad_std = float(mag_veg.std()) if mag_veg.size > 10 else 0.0

    kernel = np.ones((9, 9), np.float32) / 81.0
    mean_local = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    sq = cv2.filter2D((gray.astype(np.float32)) ** 2, -1, kernel)
    local_var = np.clip(sq - mean_local ** 2, 0, None)
    lv_veg = local_var[veg_mask > 0]
    local_var_mean = float(lv_veg.mean()) if lv_veg.size > 10 else 0.0

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_veg = lbp[veg_mask > 0]
    if lbp_veg.size > 0:
        hist, _ = np.histogram(lbp_veg, bins=10, range=(0, 10), density=True)
        lbp_entropy = float(-np.sum(hist * np.log2(hist + 1e-9)))
        lbp_std = float(lbp_veg.std())
    else:
        lbp_entropy, lbp_std = 0.0, 0.0

    hsv = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, (30, 60, 40), (90, 255, 255))
    verde = cv2.bitwise_and(verde, veg_mask)
    densidad_verde = np.count_nonzero(verde) / (np.count_nonzero(veg_mask) + 1e-6)

    return [
        lap_var, lap_var_veg, lap_abs_mean_veg,
        grad_mean, grad_std, local_var_mean,
        lbp_entropy, lbp_std, densidad_verde,
    ]


def extract_vegetation_features(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    veg_mask = _build_vegetation_mask(hsv)

    crom = _chromatic_signatures(hsv, veg_mask)
    text = _texture_features(gray, veg_mask)

    return np.array(crom + text, dtype=np.float32)


# =============================================================================
# 2. PIPELINE DE ENTRENAMIENTO
# =============================================================================
# training/train.py

def _load_dataset(dataset_dir: str):
    X, y = [], []
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

    # Iteramos sobre las carpetas mapeadas al nombre de la clase
    for class_name, folder in CLASS_FOLDERS.items():
        class_dir = os.path.join(dataset_dir, folder)
        if not os.path.isdir(class_dir):
            logger.warning(f"Carpeta no encontrada: {class_dir}")
            continue

        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(class_dir, ext)))

        if not files:
            logger.warning(f"Sin imágenes en: {class_dir}")
            continue

        logger.info(f"📂 Clase {class_name}: {len(files)} imágenes")
        for f in files:
            try:
                feats = extract_vegetation_features(f)
                X.append(feats)
                y.append(class_name)  # <-- Guardamos el string, no el ID
            except Exception as e:
                logger.error(f"✗ Error procesando {f}: {e}")

    if not X:
        raise RuntimeError("No se cargaron imágenes. Revisa el directorio del dataset.")

    return np.array(X, dtype=np.float32), np.array(y, dtype=str)

# training/train.py

def train_and_evaluate(dataset_dir: str = DATASET_DIR, save_model: bool = True):
    logger.info("=" * 70)
    logger.info("🚀 CARGANDO DATASET")
    logger.info("=" * 70)
    X, y = _load_dataset(dataset_dir)
    logger.info(f"✅ Total imágenes procesadas: {len(X)} | Vector: {X.shape[1]}D")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # Escalado
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Modelo
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    logger.info("🌲 Entrenando RandomForest...")
    model.fit(X_train_s, y_train)

    # Evaluación
    y_pred = model.predict(X_test_s)
    labels_present = sorted(np.unique(y_test))
    
    # Ya no necesitamos mapear IDs a nombres, porque "labels_present" ya contiene los strings
    target_names = labels_present 

    logger.info("\n" + "=" * 70)
    logger.info("📊 MATRIZ DE CONFUSIÓN")
    logger.info("=" * 70)
    cm = confusion_matrix(y_test, y_pred, labels=labels_present)
    logger.info(f"Etiquetas: {target_names}")
    logger.info(f"\n{cm}")

    logger.info("\n" + "=" * 70)
    logger.info("📈 CLASSIFICATION REPORT")
    logger.info("=" * 70)
    logger.info(f"\n{classification_report(y_test, y_pred, labels=labels_present, zero_division=0)}")

    accuracy = accuracy_score(y_test, y_pred)
    
    # Guardar en formato que la API espera
    if save_model:
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_path = os.path.join(SAVE_DIR, MODEL_FILENAME)
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': FEATURE_NAMES,
            'metadata': {
                'version': f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'training_date': datetime.now().isoformat(),
                'n_samples': len(X),
                'accuracy': float(accuracy),
                'classes': list(model.classes_)  # <-- Scikit-Learn guarda los strings directamente
            },
            'color_thresholds': {
                'semi_humeda':      {'hsv_lower': (35, 40, 40),  'hsv_upper': (85, 255, 255)},
                'seca':             {'hsv_lower': (20, 40, 50),  'hsv_upper': (34, 255, 230)},
                'muy_seca':         {'hsv_lower': (12, 50, 60),  'hsv_upper': (19, 255, 220)},
                'muy_seca_extrema': {'hsv_lower': (0,  40, 50),  'hsv_upper': (11, 255, 200)}
            }
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"\n💾 Modelo empaquetado y guardado en: {save_path}")

    return model, scaler


# =============================================================================
# 3. EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    train_and_evaluate(dataset_dir=DATASET_DIR, save_model=True)