"""Features module - Image transformers and estimators for the data pipeline.

This module contains all the custom transformers and estimators extracted from
the Jupyter notebook, organized into logical categories:
- Image loaders: Loading images from file paths
- Image preprocessing: Resizing, normalizing, masking, flattening, binarizing
- Image augmentation: Data augmentation techniques
- Image features: Feature extraction (histograms, PCA, standardization)
- Utilities: Visualization and saving transformers
"""

# Import tous les transformateurs depuis le module Pipelines
from .Pipelines import *
