"""Image augmentation transformers for the data pipeline."""

import random
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin


class ImageAugmenter(BaseEstimator, TransformerMixin):
    """Applique une augmentation simple (flip horizontal) à la moitié des images."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(f"\nAugmentation de {len(X)} images (flip horizontal sur 50%) ...")
        X_aug = []
        for i, img in enumerate(X):
            if i % 2 == 0:
                X_aug.append(np.fliplr(img))
            else:
                X_aug.append(img)
        print("Augmentation terminée.\n")
        return np.array(X_aug)


class ImageRandomCropper(BaseEstimator, TransformerMixin):
    """Effectue un crop aléatoire sur chaque image (carré centré ou décalé)."""

    def __init__(self, crop_size=(224, 224)):
        self.crop_size = crop_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cropped = []
        for img in tqdm(X, desc="RandomCrop"):
            h, w = img.shape[:2]
            ch, cw = self.crop_size
            if h < ch or w < cw:
                cropped.append(img)
                continue
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            cropped.append(img[top:top+ch, left:left+cw])
        print(f"Random crop terminé. Shape: {cropped[0].shape if cropped else None}")
        return np.array(cropped)
