# app/ml/feature_extractor.py
import cv2
import numpy as np
from typing import Dict, List
import logging
from skimage.feature import local_binary_pattern

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Extrae las 20 características (11 cromáticas + 9 de textura)
    usando máscara de vegetación para ignorar suelo desnudo.
    """
    
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
    
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen")
        
        # El nuevo modelo espera BGR directo y 512x512 (ver código de entrenamiento)
        img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        return img_resized

    def extract_features(self, image_bgr: np.ndarray) -> Dict[str, float]:
        """Extrae las 20 características"""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        veg_mask = self._build_vegetation_mask(hsv)
        
        crom_feats = self._chromatic_signatures(hsv, veg_mask)
        text_feats = self._texture_features(gray, veg_mask)
        
        all_feats = crom_feats + text_feats
        
        return {name: float(val) for name, val in zip(self.FEATURE_NAMES, all_feats)}
    
    def features_to_vector(self, features: Dict[str, float]) -> List[float]:
        return [features[name] for name in self.FEATURE_NAMES]

    # --- Métodos internos copiados exactamente del nuevo script ---
    
    def _build_vegetation_mask(self, hsv: np.ndarray) -> np.ndarray:
        h, s, v = cv2.split(hsv)
        mask_hue = cv2.inRange(h, 10, 95)
        mask_sat = cv2.inRange(s, 45, 255)
        mask_val = cv2.inRange(v, 30, 255)
        mask = cv2.bitwise_and(mask_hue, mask_sat)
        mask = cv2.bitwise_and(mask, mask_val)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _chromatic_signatures(self, hsv: np.ndarray, veg_mask: np.ndarray) -> list:
        h, s, v = cv2.split(hsv)
        verde = cv2.inRange(hsv, (30, 60, 40), (90, 255, 255))
        cafe1 = cv2.inRange(hsv, (5, 70, 30), (20, 255, 190))
        cafe2 = cv2.inRange(hsv, (0, 60, 30), (5, 255, 170))
        cafe  = cv2.bitwise_or(cafe1, cafe2)
        amarillo = cv2.inRange(hsv, (20, 50, 100), (32, 255, 255))

        verde    = cv2.bitwise_and(verde,    veg_mask)
        cafe     = cv2.bitwise_and(cafe,     veg_mask)
        amarillo = cv2.bitwise_and(amarillo, veg_mask)

        total_veg = int(np.count_nonzero(veg_mask)) + 1e-6

        pct_verde    = np.count_nonzero(verde)    / total_veg
        pct_cafe     = np.count_nonzero(cafe)     / total_veg
        pct_amarillo = np.count_nonzero(amarillo) / total_veg

        suma = pct_verde + pct_cafe + pct_amarillo + 1e-6
        pct_verde_n    = pct_verde    / suma
        pct_cafe_n     = pct_cafe     / suma
        pct_amarillo_n = pct_amarillo / suma

        def mean_sat(mask):
            if np.count_nonzero(mask) == 0: return 0.0
            return float(s[mask > 0].mean()) / 255.0

        sat_verde    = mean_sat(verde)
        sat_cafe     = mean_sat(cafe)
        sat_amarillo = mean_sat(amarillo)

        h_veg_vals = h[veg_mask > 0]
        if h_veg_vals.size == 0:
            mean_h, std_h = 0.0, 0.0
        else:
            mean_h = float(h_veg_vals.mean()) / 180.0
            std_h  = float(h_veg_vals.std())  / 180.0

        return [
            pct_verde, pct_cafe, pct_amarillo,
            pct_verde_n, pct_cafe_n, pct_amarillo_n,
            sat_verde, sat_cafe, sat_amarillo,
            mean_h, std_h,
        ]

    def _texture_features(self, gray: np.ndarray, veg_mask: np.ndarray) -> list:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(lap.var())

        lap_veg = lap[veg_mask > 0]
        lap_var_veg = float(lap_veg.var()) if lap_veg.size > 10 else 0.0
        lap_abs_mean_veg = float(np.abs(lap_veg).mean()) if lap_veg.size > 10 else 0.0

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        mag_veg = mag[veg_mask > 0]
        grad_mean = float(mag_veg.mean()) if mag_veg.size > 10 else 0.0
        grad_std  = float(mag_veg.std())  if mag_veg.size > 10 else 0.0

        kernel = np.ones((9, 9), np.float32) / 81.0
        mean_local = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sq = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        local_var = np.clip(sq - mean_local**2, 0, None)
        lv_veg = local_var[veg_mask > 0]
        local_var_mean = float(lv_veg.mean()) if lv_veg.size > 10 else 0.0

        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        lbp_veg = lbp[veg_mask > 0]
        if lbp_veg.size > 0:
            hist, _ = np.histogram(lbp_veg, bins=10, range=(0, 10), density=True)
            lbp_entropy = float(-np.sum(hist * np.log2(hist + 1e-9)))
            lbp_std     = float(lbp_veg.std())
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