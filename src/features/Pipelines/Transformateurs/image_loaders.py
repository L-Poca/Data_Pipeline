"""Image loading transformers for the data pipeline."""

import logging
from pathlib import Path
from typing import List, Union, Optional

from PIL import Image
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

# Configuration du logger
logger = logging.getLogger(__name__)


class ImageLoader(BaseEstimator, TransformerMixin):
    """Transformer to load images from file paths.
    
    This transformer loads images from disk and converts them to grayscale.
    It validates file paths and provides error handling for corrupted images.
    
    Parameters
    ----------
    img_size : tuple, default=(128, 128)
        Target size for images (not used in loading, kept for compatibility)
    color_mode : str, default='L'
        Color mode for PIL images ('L' for grayscale, 'RGB' for color)
    validate_paths : bool, default=True
        Whether to validate file paths before loading
    fail_on_error : bool, default=False
        If True, raises exception on loading errors. If False, skips invalid images
    verbose : bool, default=True
        Whether to display progress bar and status messages
        
    Attributes
    ----------
    n_images_loaded_ : int
        Number of successfully loaded images
    failed_images_ : list
        List of file paths that failed to load
    """

    def __init__(
        self,
        img_size: tuple = (128, 128),
        color_mode: str = 'L',
        validate_paths: bool = True,
        fail_on_error: bool = False,
        verbose: bool = True
    ):
        self.img_size = img_size
        self.color_mode = color_mode
        self.validate_paths = validate_paths
        self.fail_on_error = fail_on_error
        self.verbose = verbose

    def fit(self, data_x, data_y=None):  # pylint: disable=unused-argument
        """Fit the transformer (no-op for image loading).
        
        Parameters
        ----------
        data_x : array-like
            Input data (unused)
        data_y : array-like, optional
            Target data (unused)
            
        Returns
        -------
        self : ImageLoader
            Returns self for method chaining
        """
        return self

    def _validate_path(self, path: Union[str, Path]) -> bool:
        """Validate if path exists and is a file.
        
        Parameters
        ----------
        path : str or Path
            Path to validate
            
        Returns
        -------
        bool
            True if path is valid, False otherwise
        """
        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning(f"Path does not exist: {path}")
            return False
        if not path_obj.is_file():
            logger.warning(f"Path is not a file: {path}")
            return False
        return True

    def _load_single_image(self, path: Union[str, Path]) -> Optional[Image.Image]:
        """Load a single image from path.
        
        Parameters
        ----------
        path : str or Path
            Path to image file
            
        Returns
        -------
        PIL.Image.Image or None
            Loaded image or None if loading failed
        """
        try:
            if self.validate_paths and not self._validate_path(path):
                return None
            
            img = Image.open(path).convert(self.color_mode)
            return img
        except (IOError, OSError) as e:
            logger.error(f"Failed to load image {path}: {e}")
            if self.fail_on_error:
                raise
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading {path}: {e}")
            if self.fail_on_error:
                raise
            return None

    def transform(self, data_x: List[Union[str, Path]]) -> List[Image.Image]:
        """Transform file paths into loaded PIL images.
        
        Parameters
        ----------
        data_x : list of str or Path
            List of image file paths
            
        Returns
        -------
        list of PIL.Image.Image
            List of successfully loaded PIL Image objects
            
        Raises
        ------
        ValueError
            If no valid images could be loaded
        """
        if self.verbose:
            logger.info(f"Loading {len(data_x)} images...")
        
        images = []
        self.failed_images_ = []
        
        iterator = tqdm(data_x, desc="Loading images") if self.verbose else data_x
        
        for path in iterator:
            img = self._load_single_image(path)
            if img is not None:
                images.append(img)
            else:
                self.failed_images_.append(str(path))
        
        self.n_images_loaded_ = len(images)
        
        if self.verbose:
            success_rate = (self.n_images_loaded_ / len(data_x)) * 100
            logger.info(
                f"Loading completed: {self.n_images_loaded_}/{len(data_x)} "
                f"images loaded successfully ({success_rate:.1f}%)"
            )
            if self.failed_images_:
                logger.warning(f"Failed to load {len(self.failed_images_)} images")
        
        if not images:
            raise ValueError("No valid images could be loaded from the provided paths")
        
        return images
