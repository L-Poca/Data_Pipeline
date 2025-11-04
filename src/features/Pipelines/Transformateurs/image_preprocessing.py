"""Image preprocessing transformers for the data pipeline."""

import logging
from typing import List, Union, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

# Configuration du logger
logger = logging.getLogger(__name__)


class ImageResizer(BaseEstimator, TransformerMixin):
    """Resize images to a target size.
    
    This transformer resizes PIL Images or numpy arrays to a specified size
    using configurable interpolation methods.
    
    Parameters
    ----------
    img_size : tuple of int, default=(256, 256)
        Target size as (width, height)
    resample : int, default=Image.LANCZOS
        PIL resampling filter (NEAREST, BILINEAR, BICUBIC, LANCZOS)
    preserve_aspect_ratio : bool, default=False
        If True, resize while maintaining aspect ratio (padding if needed)
    verbose : bool, default=True
        Whether to display progress bar and status messages
        
    Attributes
    ----------
    n_images_processed_ : int
        Number of images successfully resized
    original_shapes_ : list
        Original shapes of input images
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (256, 256),
        resample: int = Image.LANCZOS,
        preserve_aspect_ratio: bool = False,
        verbose: bool = True
    ):
        self.img_size = img_size
        self.resample = resample
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.verbose = verbose

    def fit(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Fit the transformer (no-op for resizing).
        
        Parameters
        ----------
        data_x : array-like
            Input data (unused)
        data_y : array-like, optional
            Target data (unused)
            
        Returns
        -------
        self : ImageResizer
            Returns self for method chaining
        """
        return self

    def _resize_with_aspect_ratio(self, img: Image.Image) -> Image.Image:
        """Resize image while preserving aspect ratio with padding.
        
        Parameters
        ----------
        img : PIL.Image.Image
            Image to resize
            
        Returns
        -------
        PIL.Image.Image
            Resized and padded image
        """
        # Calculate scaling factor
        ratio = min(self.img_size[0] / img.size[0], self.img_size[1] / img.size[1])
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        
        # Resize image
        img_resized = img.resize(new_size, self.resample)
        
        # Create padded image
        new_img = Image.new(img.mode, self.img_size, color=0)
        paste_x = (self.img_size[0] - new_size[0]) // 2
        paste_y = (self.img_size[1] - new_size[1]) // 2
        new_img.paste(img_resized, (paste_x, paste_y))
        
        return new_img

    def transform(self, data_x: List[Union[Image.Image, np.ndarray]]) -> np.ndarray:
        """Transform images by resizing them to target size.
        
        Parameters
        ----------
        data_x : list of PIL.Image.Image or np.ndarray
            Images to resize
            
        Returns
        -------
        np.ndarray
            Resized images as numpy arrays with shape (n_samples, height, width)
            or (n_samples, height, width, channels)
        """
        if self.verbose:
            logger.info(f"Resizing {len(data_x)} images to {self.img_size}...")
        
        self.original_shapes_ = []
        resized = []
        
        iterator = tqdm(data_x, desc="Resizing images") if self.verbose else data_x
        
        for img in iterator:
            # Convert numpy array to PIL Image if needed
            if isinstance(img, np.ndarray):
                self.original_shapes_.append(img.shape)
                img = Image.fromarray(img.astype(np.uint8))
            else:
                self.original_shapes_.append(img.size)
            
            # Resize image
            if self.preserve_aspect_ratio:
                img_resized = self._resize_with_aspect_ratio(img)
            else:
                img_resized = img.resize(self.img_size, self.resample)
            
            resized.append(np.array(img_resized))
        
        self.n_images_processed_ = len(resized)
        
        if self.verbose:
            logger.info(f"Resizing completed: {self.n_images_processed_} images processed")
        
        return np.array(resized)


class ImageNormalizer(BaseEstimator, TransformerMixin):
    """Normalize image pixel values.
    
    This transformer normalizes pixel values using different strategies:
    min-max normalization, standardization, or custom range scaling.
    
    Parameters
    ----------
    method : str, default='minmax'
        Normalization method: 'minmax' (0-1), 'standard' (z-score), 
        'custom' (custom range)
    feature_range : tuple, default=(0, 1)
        Target range for 'custom' method
    per_image : bool, default=False
        If True, normalize each image independently. If False, use global stats
    verbose : bool, default=True
        Whether to display progress messages
        
    Attributes
    ----------
    global_min_ : float
        Global minimum value (when per_image=False)
    global_max_ : float
        Global maximum value (when per_image=False)
    global_mean_ : float
        Global mean value (for standardization)
    global_std_ : float
        Global standard deviation (for standardization)
    """

    def __init__(
        self,
        method: str = 'minmax',
        feature_range: Tuple[float, float] = (0, 1),
        per_image: bool = False,
        verbose: bool = True
    ):
        self.method = method
        self.feature_range = feature_range
        self.per_image = per_image
        self.verbose = verbose

    def fit(self, data_x, data_y=None):
        """Fit the transformer by computing global statistics.
        
        Parameters
        ----------
        data_x : np.ndarray
            Input images
        data_y : array-like, optional
            Target data (unused)
            
        Returns
        -------
        self : ImageNormalizer
            Returns self for method chaining
        """
        if not self.per_image:
            data_array = np.array(data_x).astype(np.float32)
            
            if self.method in ['minmax', 'custom']:
                self.global_min_ = data_array.min()
                self.global_max_ = data_array.max()
            elif self.method == 'standard':
                self.global_mean_ = data_array.mean()
                self.global_std_ = data_array.std()
                
            if self.verbose:
                logger.info(f"Fitted normalizer with method '{self.method}'")
                if self.method in ['minmax', 'custom']:
                    logger.info(f"Global range: [{self.global_min_:.2f}, {self.global_max_:.2f}]")
                elif self.method == 'standard':
                    logger.info(f"Global stats: mean={self.global_mean_:.2f}, std={self.global_std_:.2f}")
        
        return self

    def transform(self, data_x, data_y=None):
        """Transform images by normalizing pixel values.
        
        Parameters
        ----------
        data_x : np.ndarray
            Images to normalize
        data_y : array-like, optional
            Target data (unused)
            
        Returns
        -------
        np.ndarray
            Normalized images
        """
        if self.verbose:
            logger.info(f"Normalizing {len(data_x)} images using '{self.method}' method...")
        
        data_array = np.array(data_x).astype(np.float32)
        
        if self.method == 'minmax':
            if self.per_image:
                # Normalize each image to [0, 1]
                data_norm = np.zeros_like(data_array)
                for i in range(len(data_array)):
                    img_min = data_array[i].min()
                    img_max = data_array[i].max()
                    if img_max > img_min:
                        data_norm[i] = (data_array[i] - img_min) / (img_max - img_min)
                    else:
                        data_norm[i] = data_array[i]
            else:
                # Global normalization
                data_norm = (data_array - self.global_min_) / (self.global_max_ - self.global_min_)
        
        elif self.method == 'standard':
            if self.per_image:
                # Standardize each image
                data_norm = np.zeros_like(data_array)
                for i in range(len(data_array)):
                    img_mean = data_array[i].mean()
                    img_std = data_array[i].std()
                    if img_std > 0:
                        data_norm[i] = (data_array[i] - img_mean) / img_std
                    else:
                        data_norm[i] = data_array[i] - img_mean
            else:
                # Global standardization
                data_norm = (data_array - self.global_mean_) / (self.global_std_ + 1e-8)
        
        elif self.method == 'custom':
            # Scale to custom range
            if self.per_image:
                data_norm = np.zeros_like(data_array)
                for i in range(len(data_array)):
                    img_min = data_array[i].min()
                    img_max = data_array[i].max()
                    if img_max > img_min:
                        normalized = (data_array[i] - img_min) / (img_max - img_min)
                        data_norm[i] = normalized * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
                    else:
                        data_norm[i] = data_array[i]
            else:
                normalized = (data_array - self.global_min_) / (self.global_max_ - self.global_min_)
                data_norm = normalized * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        if self.verbose:
            logger.info(f"Normalization completed. Output range: [{data_norm.min():.2f}, {data_norm.max():.2f}]")
        
        return data_norm


class ImageMasker(BaseEstimator, TransformerMixin):
    """Apply binary masks to images.
    
    This transformer applies masks to images for segmentation or region isolation.
    Masks can be loaded from files or provided as arrays.
    
    Parameters
    ----------
    mask_paths : list of str or Path
        Paths to mask files corresponding to each image
    mask_threshold : float, default=0.5
        Threshold for binarizing masks (pixels > threshold become True)
    resize_masks : bool, default=True
        Whether to resize masks to match image dimensions
    invert_mask : bool, default=False
        If True, invert the mask (background becomes foreground)
    verbose : bool, default=True
        Whether to display progress bar
        
    Attributes
    ----------
    n_images_masked_ : int
        Number of images successfully masked
    """

    def __init__(
        self,
        mask_paths: List[str],
        mask_threshold: float = 0.5,
        resize_masks: bool = True,
        invert_mask: bool = False,
        verbose: bool = True
    ):
        self.mask_paths = mask_paths
        self.mask_threshold = mask_threshold
        self.resize_masks = resize_masks
        self.invert_mask = invert_mask
        self.verbose = verbose

    def fit(self, data_x, data_y=None):
        """Fit the transformer (no-op for masking).
        
        Parameters
        ----------
        data_x : array-like
            Input data (unused)
        data_y : array-like, optional
            Target data (unused)
            
        Returns
        -------
        self : ImageMasker
            Returns self for method chaining
        """
        if len(self.mask_paths) == 0:
            logger.warning("No mask paths provided")
        return self

    def transform(self, data_x: np.ndarray) -> np.ndarray:
        """Transform images by applying masks.
        
        Parameters
        ----------
        data_x : np.ndarray
            Images to mask with shape (n_samples, height, width) or 
            (n_samples, height, width, channels)
            
        Returns
        -------
        np.ndarray
            Masked images
            
        Raises
        ------
        ValueError
            If number of masks doesn't match number of images
        """
        if len(self.mask_paths) != len(data_x):
            raise ValueError(
                f"Number of masks ({len(self.mask_paths)}) must match "
                f"number of images ({len(data_x)})"
            )
        
        if self.verbose:
            logger.info(f"Applying masks to {len(data_x)} images...")
        
        masked = []
        iterator = zip(data_x, self.mask_paths)
        if self.verbose:
            iterator = tqdm(list(iterator), desc="Applying masks")
        
        for img, mask_path in iterator:
            try:
                # Load mask
                mask = Image.open(mask_path).convert('L')
                
                # Resize mask if needed
                if self.resize_masks:
                    target_size = (img.shape[1], img.shape[0])  # (width, height)
                    mask = mask.resize(target_size, Image.NEAREST)
                
                # Convert to binary array
                mask_arr = (np.array(mask) / 255.0) > self.mask_threshold
                
                # Invert if requested
                if self.invert_mask:
                    mask_arr = ~mask_arr
                
                # Apply mask
                if len(img.shape) == 3:  # Color image
                    mask_arr = mask_arr[:, :, np.newaxis]
                
                masked_img = img * mask_arr
                masked.append(masked_img)
                
            except Exception as e:
                logger.error(f"Failed to apply mask {mask_path}: {e}")
                masked.append(img)  # Use original image on error
        
        self.n_images_masked_ = len(masked)
        
        if self.verbose:
            logger.info(f"Masking completed: {self.n_images_masked_} images processed")
        
        return np.array(masked)


class ImageFlattener(BaseEstimator, TransformerMixin):
    """Flatten images for ML models.
    
    This transformer flattens 2D or 3D images into 1D feature vectors,
    suitable for traditional ML algorithms.
    
    Parameters
    ----------
    order : str, default='C'
        Flattening order: 'C' (row-major) or 'F' (column-major)
    verbose : bool, default=True
        Whether to display progress messages
        
    Attributes
    ----------
    n_features_ : int
        Number of features per image after flattening
    original_shape_ : tuple
        Original shape of images (excluding batch dimension)
    """

    def __init__(
        self,
        order: str = 'C',
        verbose: bool = True
    ):
        self.order = order
        self.verbose = verbose

    def fit(self, data_x, data_y=None):
        """Fit the transformer by storing input shape.
        
        Parameters
        ----------
        data_x : np.ndarray
            Input images
        data_y : array-like, optional
            Target data (unused)
            
        Returns
        -------
        self : ImageFlattener
            Returns self for method chaining
        """
        data_array = np.array(data_x)
        self.original_shape_ = data_array.shape[1:]
        self.n_features_ = np.prod(self.original_shape_)
        
        if self.verbose:
            logger.info(
                f"Fitted flattener: {self.original_shape_} -> {self.n_features_} features"
            )
        
        return self

    def transform(self, data_x: np.ndarray) -> np.ndarray:
        """Transform images by flattening them to 1D arrays.
        
        Parameters
        ----------
        data_x : np.ndarray
            Images to flatten with shape (n_samples, ...)
            
        Returns
        -------
        np.ndarray
            Flattened images with shape (n_samples, n_features)
        """
        if self.verbose:
            logger.info(f"Flattening {len(data_x)} images...")
        
        data_array = np.array(data_x)
        n_samples = data_array.shape[0]
        
        # Flatten each image
        data_flat = data_array.reshape(n_samples, -1, order=self.order)
        
        if self.verbose:
            logger.info(
                f"Flattening completed: {data_array.shape} -> {data_flat.shape}"
            )
        
        return data_flat
    
    def inverse_transform(self, data_x: np.ndarray) -> np.ndarray:
        """Reshape flattened images back to original shape.
        
        Parameters
        ----------
        data_x : np.ndarray
            Flattened images with shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Reshaped images with original dimensions
        """
        if not hasattr(self, 'original_shape_'):
            raise RuntimeError("Transformer must be fitted before inverse_transform")
        
        n_samples = data_x.shape[0]
        target_shape = (n_samples,) + self.original_shape_
        
        return data_x.reshape(target_shape, order=self.order)


class ImageBinarizer(BaseEstimator, TransformerMixin):
    """Binarize images using threshold.
    
    This transformer applies binary thresholding to images, converting
    them to binary (0/1) values. Supports multiple thresholding methods.
    
    Parameters
    ----------
    threshold : float or str, default=0.5
        Threshold value or method. If float, uses fixed threshold.
        If 'otsu', uses Otsu's method. If 'mean', uses mean value.
        If 'median', uses median value.
    invert : bool, default=False
        If True, invert the binary result (values > threshold become 0)
    output_dtype : type, default=np.float32
        Data type of output array
    verbose : bool, default=True
        Whether to display progress messages
        
    Attributes
    ----------
    threshold_value_ : float
        Actual threshold value used (computed for adaptive methods)
    """

    def __init__(
        self,
        threshold: Union[float, str] = 0.5,
        invert: bool = False,
        output_dtype=np.float32,
        verbose: bool = True
    ):
        self.threshold = threshold
        self.invert = invert
        self.output_dtype = output_dtype
        self.verbose = verbose

    def fit(self, data_x, data_y=None):
        """Fit the transformer by computing threshold if needed.
        
        Parameters
        ----------
        data_x : np.ndarray
            Input images
        data_y : array-like, optional
            Target data (unused)
            
        Returns
        -------
        self : ImageBinarizer
            Returns self for method chaining
        """
        data_array = np.array(data_x)
        
        if isinstance(self.threshold, str):
            if self.threshold == 'mean':
                self.threshold_value_ = data_array.mean()
            elif self.threshold == 'median':
                self.threshold_value_ = np.median(data_array)
            elif self.threshold == 'otsu':
                # Simple Otsu implementation
                hist, bin_edges = np.histogram(data_array.flatten(), bins=256)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Compute optimal threshold
                max_var = 0
                optimal_threshold = 0
                
                for t_idx in range(1, len(hist)-1):
                    w0 = hist[:t_idx].sum()
                    w1 = hist[t_idx:].sum()
                    
                    if w0 == 0 or w1 == 0:
                        continue
                    
                    mu0 = (hist[:t_idx] * bin_centers[:t_idx]).sum() / w0
                    mu1 = (hist[t_idx:] * bin_centers[t_idx:]).sum() / w1
                    
                    var = w0 * w1 * (mu0 - mu1) ** 2
                    
                    if var > max_var:
                        max_var = var
                        optimal_threshold = bin_centers[t_idx]
                
                self.threshold_value_ = optimal_threshold
            else:
                raise ValueError(f"Unknown threshold method: {self.threshold}")
        else:
            self.threshold_value_ = float(self.threshold)
        
        if self.verbose:
            logger.info(f"Fitted binarizer with threshold: {self.threshold_value_:.4f}")
        
        return self

    def transform(self, data_x, data_y=None):
        """Transform images by applying binary thresholding.
        
        Parameters
        ----------
        data_x : np.ndarray
            Images to binarize
        data_y : array-like, optional
            Target data (unused)
            
        Returns
        -------
        np.ndarray
            Binarized images
        """
        if not hasattr(self, 'threshold_value_'):
            # If not fitted, use threshold as-is
            if isinstance(self.threshold, str):
                raise RuntimeError("Transformer must be fitted for adaptive thresholding")
            self.threshold_value_ = float(self.threshold)
        
        if self.verbose:
            logger.info(
                f"Binarizing {len(data_x)} images with threshold {self.threshold_value_:.4f}"
            )
        
        data_array = np.array(data_x)
        
        if self.invert:
            binarized = (data_array <= self.threshold_value_).astype(self.output_dtype)
        else:
            binarized = (data_array > self.threshold_value_).astype(self.output_dtype)
        
        if self.verbose:
            positive_ratio = (binarized == 1).mean() * 100
            logger.info(f"Binarization completed. Positive pixels: {positive_ratio:.1f}%")
        
        return binarized
