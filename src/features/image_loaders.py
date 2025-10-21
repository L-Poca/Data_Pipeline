"""Image loading transformers for the data pipeline."""

from PIL import Image
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin


class ImageLoader(BaseEstimator, TransformerMixin):
    """Transformer pour charger et prétraiter les images."""

    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"\nChargement de {len(X)} images...")
        images = []
        for path in tqdm(X, desc="Images"):
            img = Image.open(path).convert('L')  # PAS de resize ici
            images.append(img)
        print("Chargement terminé.\n")
        return images  # Liste d'images PIL
