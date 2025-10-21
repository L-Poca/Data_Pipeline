"""Transformateurs pour les pipelines d'images.

Ce module contient tous les transformateurs personnalisés pour le traitement
des images médicales, organisés par catégorie :
- Image loaders: Chargement des images depuis les chemins
- Image preprocessing: Redimensionnement, normalisation, masquage, aplatissement, binarisation
- Image augmentation: Techniques d'augmentation des données
- Image features: Extraction de caractéristiques (histogrammes, PCA, standardisation)
- Utilities: Transformateurs de visualisation et sauvegarde
"""

# Image loaders
from .image_loaders import ImageLoader

# Image preprocessing
from .image_preprocessing import (
    ImageResizer,
    ImageNormalizer,
    ImageMasker,
    ImageFlattener,
    ImageBinarizer,
)

# Image augmentation
from .image_augmentation import (
    ImageAugmenter,
    ImageRandomCropper,
)

# Image features
from .image_features import (
    ImageHistogram,
    ImagePCA,
    ImageStandardScaler,
)

# Utilities
from .utilities import (
    VisualizeTransformer,
    SaveTransformer,
)

__all__ = [
    # Loaders
    "ImageLoader",
    # Preprocessing
    "ImageResizer",
    "ImageNormalizer",
    "ImageMasker",
    "ImageFlattener",
    "ImageBinarizer",
    # Augmentation
    "ImageAugmenter",
    "ImageRandomCropper",
    # Features
    "ImageHistogram",
    "ImagePCA",
    "ImageStandardScaler",
    # Utilities
    "VisualizeTransformer",
    "SaveTransformer",
]