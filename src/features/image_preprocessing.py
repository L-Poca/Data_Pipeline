"""Image preprocessing transformers for the data pipeline."""

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin


class ImageResizer(BaseEstimator, TransformerMixin):
    """Redimensionne les images PIL ou numpy array à la taille souhaitée."""

    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"\nRedimensionnement de {len(X)} images en {self.img_size} ...")
        resized = []
        for img in tqdm(X, desc="Redimensionnement"):
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_resized = img.resize(self.img_size)
            resized.append(np.array(img_resized))
        print("Redimensionnement terminé.\n")
        return np.array(resized)


class ImageNormalizer(BaseEstimator, TransformerMixin):
    """Normalise les images (array) pixel-wise entre 0 et 1."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(f"\nNormalisation de {len(X)} images ...")
        X_norm = np.array(X).astype(np.float32) / 255.0
        print("Normalisation terminée.\n")
        return X_norm


class ImageMasker(BaseEstimator, TransformerMixin):
    """Applique des masques sur les images."""

    def __init__(self, mask_paths):
        self.mask_paths = mask_paths

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"\nApplication des masques sur {len(X)} images ...")
        masked = []
        for img, mask_path in tqdm(zip(X, self.mask_paths), desc="Masquage", total=len(X)):
            mask = Image.open(mask_path).convert('L').resize(img.shape[::-1])
            mask_arr = np.array(mask) > 0  # binaire
            masked.append(img * mask_arr)
        print("Masquage terminé.\n")
        return np.array(masked)


class ImageFlattener(BaseEstimator, TransformerMixin):
    """Aplatit les images pour les modèles ML."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"\nAplatissement de {X.shape[0]} images ...")
        X_flat = []
        for img in tqdm(X, desc="Aplatissement"):
            X_flat.append(img.flatten())
        X_flat = np.array(X_flat)
        print("Aplatissement terminé.\n")
        return X_flat


class ImageBinarizer(BaseEstimator, TransformerMixin):
    """Binarise les images (seuil fixe ou automatique)."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(f"Binarisation avec seuil {self.threshold}")
        return (X > self.threshold).astype(np.float32)
