"""Image feature extraction transformers for the data pipeline."""

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuration du logger
logger = logging.getLogger(__name__)


class ImageHistogram(BaseEstimator, TransformerMixin):
    """Extract histogram features from images.
    
    This transformer computes intensity histograms for each image, which can
    be used as features for machine learning models.
    
    Parameters
    ----------
    bins : int, default=32
        Number of bins for histogram computation
    range : tuple or None, default=None
        Range for histogram bins as (min, max). If None, uses data range
    density : bool, default=False
        If True, compute normalized histogram (probability density)
    per_channel : bool, default=False
        If True and images are multi-channel, compute histogram per channel
    verbose : bool, default=True
        Whether to display progress messages
        
    Attributes
    ----------
    n_features_ : int
        Number of histogram features per image
    """

    def __init__(
        self,
        bins: int = 32,
        range: Optional[Tuple[float, float]] = None,
        density: bool = False,
        per_channel: bool = False,
        verbose: bool = True
    ):
        self.bins = bins
        self.range = range
        self.density = density
        self.per_channel = per_channel
        self.verbose = verbose

    def fit(self, data_x, data_y=None):
        """Fit the transformer by determining feature dimensions.
        
        Parameters
        ----------
        data_x : np.ndarray
            Input images
        data_y : array-like, optional
            Target data (unused)
            
        Returns
        -------
        self : ImageHistogram
            Returns self for method chaining
        """
        data_array = np.array(data_x)
        
        # Determine range if not provided
        if self.range is None:
            self.range_ = (data_array.min(), data_array.max())
        else:
            self.range_ = self.range
        
        # Determine number of features
        if self.per_channel and len(data_array.shape) == 4:
            n_channels = data_array.shape[-1]
            self.n_features_ = self.bins * n_channels
        else:
            self.n_features_ = self.bins
        
        if self.verbose:
            logger.info(
                f"Fitted histogram extractor: {self.bins} bins, "
                f"range={self.range_}, {self.n_features_} features"
            )
        
        return self

    def transform(self, data_x, data_y=None):
        """Transform images by computing histograms.
        
        Parameters
        ----------
        data_x : np.ndarray
            Images to transform
        data_y : array-like, optional
            Target data (unused)
            
        Returns
        -------
        np.ndarray
            Histogram features with shape (n_samples, n_features)
        """
        if not hasattr(self, 'range_'):
            self.fit(data_x)
        
        if self.verbose:
            logger.info(f"Computing histograms for {len(data_x)} images...")
        
        data_array = np.array(data_x)
        histograms = []
        
        for img in data_array:
            if self.per_channel and len(img.shape) == 3:
                # Compute histogram per channel and concatenate
                hist_channels = []
                for c in range(img.shape[-1]):
                    hist, _ = np.histogram(
                        img[:, :, c].flatten(),
                        bins=self.bins,
                        range=self.range_,
                        density=self.density
                    )
                    hist_channels.append(hist)
                histogram = np.concatenate(hist_channels)
            else:
                # Compute single histogram
                histogram, _ = np.histogram(
                    img.flatten(),
                    bins=self.bins,
                    range=self.range_,
                    density=self.density
                )
            
            histograms.append(histogram)
        
        result = np.array(histograms)
        
        if self.verbose:
            logger.info(f"Histogram extraction completed. Shape: {result.shape}")
        
        return result


class ImagePCA(BaseEstimator, TransformerMixin):
    """Apply PCA dimensionality reduction to images.
    
    This transformer flattens images and applies Principal Component Analysis
    to reduce dimensionality while preserving most variance.
    
    Parameters
    ----------
    n_components : int or float, default=50
        Number of components to keep. If float (0 < n_components < 1),
        select the number of components such that the explained variance
        is greater than the specified percentage
    whiten : bool, default=False
        If True, components are multiplied by sqrt(n_samples) and divided
        by singular values to ensure uncorrelated outputs with unit variance
    svd_solver : str, default='auto'
        SVD solver: 'auto', 'full', 'arpack', 'randomized'
    random_state : int or None, default=None
        Random state for reproducibility
    verbose : bool, default=True
        Whether to display progress messages
        
    Attributes
    ----------
    pca_ : PCA
        Fitted PCA object
    n_components_ : int
        Actual number of components used
    explained_variance_ratio_ : float
        Cumulative explained variance ratio
    """

    def __init__(
        self,
        n_components: int = 50,
        whiten: bool = False,
        svd_solver: str = 'auto',
        random_state: Optional[int] = None,
        verbose: bool = True
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, data_x, data_y=None):
        """Fit PCA on flattened images.
        
        Parameters
        ----------
        data_x : np.ndarray
            Input images
        data_y : array-like, optional
            Target data (unused)
            
        Returns
        -------
        self : ImagePCA
            Returns self for method chaining
        """
        data_array = np.array(data_x)
        n_samples = data_array.shape[0]
        data_flat = data_array.reshape(n_samples, -1)
        
        if self.verbose:
            logger.info(
                f"Fitting PCA on {n_samples} images with shape {data_array.shape[1:]}..."
            )
        
        self.pca_ = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            random_state=self.random_state
        )
        self.pca_.fit(data_flat)
        
        self.n_components_ = self.pca_.n_components_
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_.sum()
        
        if self.verbose:
            logger.info(
                f"PCA fitted: {self.n_components_} components, "
                f"explained variance: {self.explained_variance_ratio_:.2%}"
            )
        
        return self

    def transform(self, data_x, data_y=None):
        """Transform images using fitted PCA.
        
        Parameters
        ----------
        data_x : np.ndarray
            Images to transform
        data_y : array-like, optional
            Target data (unused)
            
        Returns
        -------
        np.ndarray
            PCA-transformed features with shape (n_samples, n_components)
        """
        if not hasattr(self, 'pca_'):
            raise RuntimeError("PCA must be fitted before transform")
        
        data_array = np.array(data_x)
        n_samples = data_array.shape[0]
        data_flat = data_array.reshape(n_samples, -1)
        
        if self.verbose:
            logger.info(f"Transforming {n_samples} images with PCA...")
        
        data_pca = self.pca_.transform(data_flat) # methode transform de PCA
        
        if self.verbose:
            logger.info(f"PCA transformation completed. Shape: {data_pca.shape}")
        
        return data_pca
    
    def inverse_transform(self, data_x: np.ndarray) -> np.ndarray:
        """Reconstruct images from PCA components.
        
        Parameters
        ----------
        data_x : np.ndarray
            PCA-transformed data
            
        Returns
        -------
        np.ndarray
            Reconstructed flattened images
        """
        if not hasattr(self, 'pca_'):
            raise RuntimeError("PCA must be fitted before inverse_transform")
        
        return self.pca_.inverse_transform(data_x) # methode inverse_transform de PCA


class ImageStandardScaler(BaseEstimator, TransformerMixin):
    """Apply StandardScaler to images.
    
    This transformer flattens images, applies standardization (z-score),
    and optionally reshapes back to original image dimensions.
    
    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling
    with_std : bool, default=True
        If True, scale the data to unit variance
    reshape_output : bool, default=True
        If True, reshape output to original image dimensions
    verbose : bool, default=True
        Whether to display progress messages
        
    Attributes
    ----------
    scaler_ : StandardScaler
        Fitted StandardScaler object
    original_shape_ : tuple
        Original shape of images (excluding batch dimension)
    """

    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        reshape_output: bool = True,
        verbose: bool = True
    ):
        self.with_mean = with_mean
        self.with_std = with_std
        self.reshape_output = reshape_output
        self.verbose = verbose

    def fit(self, data_x, data_y=None):
        """Fit StandardScaler on flattened images.
        
        Parameters
        ----------
        data_x : np.ndarray
            Input images
        data_y : array-like, optional
            Target data (unused)
            
        Returns
        -------
        self : ImageStandardScaler
            Returns self for method chaining
        """
        data_array = np.array(data_x)
        self.original_shape_ = data_array.shape[1:]
        
        n_samples = data_array.shape[0]
        data_flat = data_array.reshape(n_samples, -1)
        
        if self.verbose:
            logger.info(f"Fitting StandardScaler on {n_samples} flattened images...")
        
        self.scaler_ = StandardScaler(
            with_mean=self.with_mean,
            with_std=self.with_std
        )
        self.scaler_.fit(data_flat) # methode fit de StandardScaler
        
        if self.verbose:
            logger.info(
                f"StandardScaler fitted. Mean: {self.scaler_.mean_.mean():.4f}, "
                f"Std: {self.scaler_.scale_.mean():.4f}"
            )
        
        return self

    def transform(self, data_x, data_y=None):
        """Transform images using fitted StandardScaler.
        
        Parameters
        ----------
        data_x : np.ndarray
            Images to transform
        data_y : array-like, optional
            Target data (unused)
            
        Returns
        -------
        np.ndarray
            Standardized images. If reshape_output=True, maintains original
            image shape. Otherwise, returns flattened arrays.
        """
        if not hasattr(self, 'scaler_'):
            raise RuntimeError("StandardScaler must be fitted before transform")
        
        data_array = np.array(data_x)
        n_samples = data_array.shape[0]
        data_flat = data_array.reshape(n_samples, -1)
        
        if self.verbose:
            logger.info(f"Standardizing {n_samples} images...")
        
        data_scaled = self.scaler_.transform(data_flat)
        
        # Reshape back to original dimensions if requested
        if self.reshape_output:
            data_scaled = data_scaled.reshape(data_array.shape)
        
        if self.verbose:
            logger.info(
                f"Standardization completed. Shape: {data_scaled.shape}, "
                f"Mean: {data_scaled.mean():.4f}, Std: {data_scaled.std():.4f}"
            )
        
        return data_scaled
    
    def inverse_transform(self, data_x: np.ndarray) -> np.ndarray:
        """Reverse standardization.
        
        Parameters
        ----------
        data_x : np.ndarray
            Standardized data
            
        Returns
        -------
        np.ndarray
            Original scale data
        """
        if not hasattr(self, 'scaler_'):
            raise RuntimeError("StandardScaler must be fitted before inverse_transform")
        
        original_shape = data_x.shape
        
        # Flatten if needed
        if len(original_shape) > 2:
            n_samples = original_shape[0]
            data_flat = data_x.reshape(n_samples, -1)
        else:
            data_flat = data_x
        
        # Inverse transform
        data_inverse = self.scaler_.inverse_transform(data_flat)
        
        # Reshape back if needed
        if len(original_shape) > 2 and self.reshape_output:
            data_inverse = data_inverse.reshape(original_shape)
        
        return data_inverse
