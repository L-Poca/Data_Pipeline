"""Transformateurs pour le traitement d'images médicales."""

# Image loaders
from .image_loaders import (
    ImageLoader,
)

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
