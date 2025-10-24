"""Image loading transformers for the data pipeline."""

from PIL import Image
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin


class ImageLoader(BaseEstimator, TransformerMixin):
    """Transformer pour charger et prétraiter les images."""

    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size

    def fit(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Fit the transformer (no-op for image loading).
        
        Args:
            data_x: Input data (unused)
            data_y: Target data (unused)
            
        Returns:
            self: Returns self for method chaining
        """
        return self

    def transform(self, data_x):
        """Transform file paths into loaded PIL images.
        
        Args:
            data_x: List of image file paths
            
        Returns:
            list: List of PIL Image objects
        """
        print(f"\nChargement de {len(data_x)} images...")
        images = []
        for path in tqdm(data_x, desc="Images"):
            img = Image.open(path).convert('L')  # PAS de resize ici
            images.append(img)
        print("Chargement terminé.\n")
        return images  # Liste d'images PIL
