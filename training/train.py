# training/train.py
import warnings
import joblib
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self, save_dir="models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.model = None
        self.metadata = {}

    def generate_training_data(self, n_samples=4000):
        """
        Genera datos sintéticos para 4 categorías con distribuciones
        mejor calibradas según las imágenes reales de referencia.
        """
        np.random.seed(42)
        X = []
        y = []

        # Parámetros por clase (media, std) calibrados con imágenes reales
        # semi_humeda      -> bosque denso verde (imagen 4)
        # seca             -> matorral mixto verde/marrón (imagen 1)
        # muy_seca         -> pastizal dorado seco (imagen 2)
        # muy_seca_extrema -> suelo árido, ramas secas (imagen 3)
        class_params = {
            'semi_humeda': {
                'green': (0.75, 0.10),
                'brown': (0.10, 0.05),
                'ndvi':  (0.65, 0.15),
                'gci':   (0.60, 0.15),
                'hue':   (90, 15),
                'sat':   (180, 30),
                'val':   (140, 30)
            },
            'seca': {
                'green': (0.30, 0.15),
                'brown': (0.45, 0.15),
                'ndvi':  (0.10, 0.20),
                'gci':   (0.05, 0.20),
                'hue':   (40, 15),
                'sat':   (110, 35),
                'val':   (130, 35)
            },
            'muy_seca': {
                'green': (0.10, 0.08),
                'brown': (0.70, 0.15),
                'ndvi':  (-0.25, 0.15),
                'gci':   (-0.25, 0.15),
                'hue':   (25, 10),
                'sat':   (90, 30),
                'val':   (150, 35)
            },
            'muy_seca_extrema': {
                'green': (0.02, 0.03),
                'brown': (0.85, 0.10),
                'ndvi':  (-0.55, 0.12),
                'gci':   (-0.55, 0.12),
                'hue':   (10, 8),
                'sat':   (70, 25),
                'val':   (160, 30)
            }
        }

        samples_per_class = n_samples // 4

        for cls_name, params in class_params.items():
            for _ in range(samples_per_class):
                green = np.clip(np.random.normal(*params['green']), 0, 1)
                brown = np.clip(np.random.normal(*params['brown']), 0, 1)
                ndvi = np.clip(np.random.normal(*params['ndvi']), -1, 1)
                gci = np.clip(np.random.normal(*params['gci']), -1, 1)
                hue = np.clip(np.random.normal(*params['hue']), 0, 179)
                sat = np.clip(np.random.normal(*params['sat']), 0, 255)
                val = np.clip(np.random.normal(*params['val']), 0, 255)

                # Features derivadas (más discriminativas)
                green_brown_ratio = green / (brown + 1e-6)
                green_minus_brown = green - brown
                ndvi_gci_inter = ndvi * gci
                hue_dist_green = np.abs(hue - 60) / 60.0

                X.append([
                    green,
                    brown,
                    ndvi,
                    gci,
                    hue,
                    sat,
                    val,
                    green_brown_ratio,
                    green_minus_brown,
                    ndvi_gci_inter,
                    hue_dist_green
                ])
                y.append(cls_name)

        return np.array(X), np.array(y)

    def train(self, X, y, test_size=0.2, random_state=42):
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state,
            stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=1,
            class_weight='balanced',   # CLAVE: balancea clases minoritarias
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Validación cruzada con F1 macro (más robusto ante desbalance)
        cv_scores = cross_val_score(
            self.model, X_scaled, y, cv=5, scoring='f1_macro'
        )

        feature_names = [
            'green_pixel_ratio', 'brown_pixel_ratio',
            'ndvi_simulated', 'gci', 'hsv_mean_h',
            'hsv_mean_s', 'hsv_mean_v',
            'green_brown_ratio', 'green_minus_brown',
            'ndvi_gci_interaction', 'hue_distance_to_green'
        ]

        report_dict = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        self.metadata = {
            'version': f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'training_date': datetime.now().isoformat(),
            'n_samples': len(X),
            'test_size': test_size,
            'accuracy': accuracy,
            'f1_macro_cv_mean': float(cv_scores.mean()),
            'f1_macro_cv_std': float(cv_scores.std()),
            'feature_names': feature_names,
            'classes': list(self.model.classes_),
            'classification_report': report_dict
        }

        print("\n" + "=" * 60)
        print("REPORTE DE ENTRENAMIENTO (4 CATEGORÍAS)")
        print("=" * 60)
        print(f"Muestras totales: {len(X)}")
        print(f"Precisión en test: {accuracy:.3f}")
        print(
            f"F1 macro (CV 5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print("\nReporte por clase:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("=" * 60)

        return self.model

    def save_model(self, filename="vegetation_classifier"):
        version = self.metadata['version']
        filepath = os.path.join(self.save_dir, f"{filename}_{version}.joblib")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.metadata['feature_names'],
            'metadata': self.metadata,
            # Umbrales HSV alineados con las clases reales
            'color_thresholds': {
                'semi_humeda':      {'hsv_lower': (35, 40, 40),  'hsv_upper': (85, 255, 255)},
                'seca':             {'hsv_lower': (20, 40, 50),  'hsv_upper': (34, 255, 230)},
                'muy_seca':         {'hsv_lower': (12, 50, 60),  'hsv_upper': (19, 255, 220)},
                'muy_seca_extrema': {'hsv_lower': (0,  40, 50),  'hsv_upper': (11, 255, 200)}
            }
        }

        joblib.dump(model_data, filepath)
        print(f"\n✅ Modelo guardado en: {filepath}")

        latest_path = os.path.join(
            self.save_dir, "vegetation_classifier_latest.joblib")
        joblib.dump(model_data, latest_path)
        print(f"✅ Último modelo guardado en: {latest_path}")

        return filepath


def main():
    print("=" * 60)
    print("ENTRENAMIENTO DEL CLASIFICADOR DE VEGETACIÓN (4 CATEGORÍAS)")
    print("=" * 60)

    trainer = ModelTrainer()

    print("\n⏳ Generando datos de entrenamiento...")
    X, y = trainer.generate_training_data(n_samples=4000)
    print(f"✅ Generadas {len(X)} muestras")
    print(f"   Distribución: {dict(zip(*np.unique(y, return_counts=True)))}")

    print("\n⏳ Entrenando modelo...")
    trainer.train(X, y)

    print("\n⏳ Guardando modelo...")
    trainer.save_model()

    print("\n✅ Entrenamiento completado!")


if __name__ == "__main__":
    main()
