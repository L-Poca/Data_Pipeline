"""Image feature extraction transformers for the data pipeline."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ImageHistogram(BaseEstimator, TransformerMixin):
    """Calcule l'histogramme d'intensité pour chaque image."""

    def __init__(self, bins=32):
        self.bins = bins

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(f"Calcul des histogrammes ({self.bins} bins)")
        histos = [np.histogram(img.flatten(), bins=self.bins, range=(0, 1))[0] for img in X]
        return np.array(histos)


class ImagePCA(BaseEstimator, TransformerMixin):
    """Réduction de dimension par ACP (PCA) sur les images aplaties."""

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y=None):
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        self.pca.fit(X_flat)
        return self

    def transform(self, X, y=None):
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        X_pca = self.pca.transform(X_flat)
        print(f"PCA terminé. Shape: {X_pca.shape}")
        return X_pca


class ImageStandardScaler(BaseEstimator, TransformerMixin):
    """Applique un StandardScaler pixel-wise sur les images aplaties."""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        self.scaler.fit(X_flat)
        return self

    def transform(self, X, y=None):
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        X_scaled = self.scaler.transform(X_flat)
        print(f"Standardisation terminée. Shape: {X_scaled.shape}")
        return X_scaled.reshape(X.shape)
