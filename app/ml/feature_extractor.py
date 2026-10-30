# app/ml/feature_extractor.py
import cv2
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extrae las 11 características usadas por el modelo entrenado.

    Features base (7):
        - green_pixel_ratio
        - brown_pixel_ratio
        - ndvi_simulated
        - gci
        - hsv_mean_h
        - hsv_mean_s
        - hsv_mean_v

    Features derivadas (4):
        - green_brown_ratio
        - green_minus_brown
        - ndvi_gci_interaction
        - hue_distance_to_green
    """

    # Orden EXACTO de features (debe coincidir con training/train.py)
    FEATURE_NAMES = [
        'green_pixel_ratio',
        'brown_pixel_ratio',
        'ndvi_simulated',
        'gci',
        'hsv_mean_h',
        'hsv_mean_s',
        'hsv_mean_v',
        'green_brown_ratio',
        'green_minus_brown',
        'ndvi_gci_interaction',
        'hue_distance_to_green'
    ]

    def __init__(self, target_size=(400, 400)):
        self.target_size = target_size

        # Umbrales HSV alineados con las imágenes reales de referencia
        self.color_thresholds = {
            'semi_humeda':      {'hsv_lower': np.array([35, 40, 40]),  'hsv_upper': np.array([85, 255, 255])},
            'seca':             {'hsv_lower': np.array([20, 40, 50]),  'hsv_upper': np.array([34, 255, 230])},
            'muy_seca':         {'hsv_lower': np.array([12, 50, 60]),  'hsv_upper': np.array([19, 255, 220])},
            'muy_seca_extrema': {'hsv_lower': np.array([0,  40, 50]),  'hsv_upper': np.array([11, 255, 200])}
        }

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("No se pudo decodificar la imagen")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.target_size)
        return img_resized

    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extrae las 11 características en el orden esperado por el modelo."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # --- Features base ---
        green_pixel_ratio = self._green_pixel_ratio(hsv)
        brown_pixel_ratio = self._brown_pixel_ratio(hsv)
        ndvi_sim = self._calculate_simulated_ndvi(image)
        gci = self._calculate_gci(image)

        hsv_mean = np.mean(hsv, axis=(0, 1))
        hsv_mean_h = float(hsv_mean[0])
        hsv_mean_s = float(hsv_mean[1])
        hsv_mean_v = float(hsv_mean[2])

        # --- Features derivadas (deben coincidir con train.py) ---
        green_brown_ratio = green_pixel_ratio / (brown_pixel_ratio + 1e-6)
        green_minus_brown = green_pixel_ratio - brown_pixel_ratio
        ndvi_gci_inter = ndvi_sim * gci
        hue_dist_green = abs(hsv_mean_h - 60) / 60.0

        return {
            'green_pixel_ratio':    green_pixel_ratio,
            'brown_pixel_ratio':    brown_pixel_ratio,
            'ndvi_simulated':       ndvi_sim,
            'gci':                  gci,
            'hsv_mean_h':           hsv_mean_h,
            'hsv_mean_s':           hsv_mean_s,
            'hsv_mean_v':           hsv_mean_v,
            'green_brown_ratio':    green_brown_ratio,
            'green_minus_brown':    green_minus_brown,
            'ndvi_gci_interaction': ndvi_gci_inter,
            'hue_distance_to_green': hue_dist_green
        }

    def features_to_vector(self, features: Dict[str, float]) -> List[float]:
        """Devuelve el vector en el ORDEN exacto del entrenamiento."""
        return [float(features[name]) for name in self.FEATURE_NAMES]

    # ---------- Cálculos internos ----------

    def _green_pixel_ratio(self, hsv_image: np.ndarray) -> float:
        lower = self.color_thresholds['semi_humeda']['hsv_lower']
        upper = self.color_thresholds['semi_humeda']['hsv_upper']
        mask = cv2.inRange(hsv_image, lower, upper)
        return float(np.sum(mask > 0) / mask.size)

    def _brown_pixel_ratio(self, hsv_image: np.ndarray) -> float:
        # Cubre seca + muy_seca + muy_seca_extrema
        lower = np.array([0, 40, 50])
        upper = np.array([34, 255, 230])
        mask = cv2.inRange(hsv_image, lower, upper)
        return float(np.sum(mask > 0) / mask.size)

    def _calculate_simulated_ndvi(self, image: np.ndarray) -> float:
        nir_sim = image[:, :, 1].astype(np.float32)
        red = image[:, :, 0].astype(np.float32)
        ndvi = (nir_sim - red) / (nir_sim + red + 1e-6)
        return float(np.mean(ndvi))

    def _calculate_gci(self, image: np.ndarray) -> float:
        green = image[:, :, 1].astype(np.float32)
        red = image[:, :, 0].astype(np.float32)
        blue = image[:, :, 2].astype(np.float32)
        gci = (2 * green - red - blue) / (2 * green + red + blue + 1e-6)
        return float(np.mean(gci))
