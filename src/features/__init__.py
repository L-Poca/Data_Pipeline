"""Features module - Image transformers and estimators for the data pipeline.

This module contains all the custom transformers and estimators extracted from
the Jupyter notebook, organized into logical categories:
- Image loaders: Loading images from file paths
- Image preprocessing: Resizing, normalizing, masking, flattening, binarizing
- Image augmentation: Data augmentation techniques
- Image features: Feature extraction (histograms, PCA, standardization)
- Utilities: Visualization and saving transformers
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
