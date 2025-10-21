# Migration Guide: Using the Features Module

This guide explains how to migrate from using transformers defined in the notebook to using the new `src.features` module.

## What Changed?

All 13 custom transformers and estimators have been extracted from the Jupyter notebook (`src/notebooks/main_pipeline.ipynb`) and organized into the `src/features/` module.

## Before (Notebook)

Previously, you had to define or copy-paste transformer classes in your notebook:

```python
# In notebook - transformers defined inline
class ImageLoader(BaseEstimator, TransformerMixin):
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
    # ... rest of implementation

class ImageResizer(BaseEstimator, TransformerMixin):
    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size
    # ... rest of implementation

# Create pipeline
pipeline = Pipeline([
    ('loader', ImageLoader()),
    ('resizer', ImageResizer()),
])
```

## After (Using Features Module)

Now you can simply import the transformers from the features module:

```python
# Import from features module
from src.features import ImageLoader, ImageResizer, ImageNormalizer, ImageFlattener
from sklearn.pipeline import Pipeline

# Create pipeline - same as before!
pipeline = Pipeline([
    ('loader', ImageLoader()),
    ('resizer', ImageResizer()),
    ('normalizer', ImageNormalizer()),
    ('flattener', ImageFlattener()),
])
```

## Available Transformers

All transformers are organized by functionality:

### Image Loading
```python
from src.features import ImageLoader
```

### Preprocessing
```python
from src.features import (
    ImageResizer,
    ImageNormalizer,
    ImageMasker,
    ImageFlattener,
    ImageBinarizer,
)
```

### Augmentation
```python
from src.features import (
    ImageAugmenter,
    ImageRandomCropper,
)
```

### Feature Extraction
```python
from src.features import (
    ImageHistogram,
    ImagePCA,
    ImageStandardScaler,
)
```

### Utilities
```python
from src.features import (
    VisualizeTransformer,
    SaveTransformer,
)
```

## Complete Example

Here's a complete example of migrating a pipeline:

### Before (In Notebook)
```python
# Cell 1: Define all transformers (many cells of code)
class ImageLoader(BaseEstimator, TransformerMixin):
    # ... 20+ lines of code

class ImageResizer(BaseEstimator, TransformerMixin):
    # ... 20+ lines of code

# Cell 2: Create pipeline
pipeline = Pipeline([
    ('loader', ImageLoader(img_size=(128, 128))),
    ('resizer', ImageResizer(img_size=(256, 256))),
    ('normalizer', ImageNormalizer()),
    ('augmenter', ImageAugmenter()),
    ('flattener', ImageFlattener()),
])

# Cell 3: Use pipeline
X_transformed = pipeline.fit_transform(image_paths, y)
```

### After (In Notebook)
```python
# Cell 1: Import transformers - one line!
from src.features import (
    ImageLoader, ImageResizer, ImageNormalizer,
    ImageAugmenter, ImageFlattener
)
from sklearn.pipeline import Pipeline

# Cell 2: Create pipeline - same as before
pipeline = Pipeline([
    ('loader', ImageLoader(img_size=(128, 128))),
    ('resizer', ImageResizer(img_size=(256, 256))),
    ('normalizer', ImageNormalizer()),
    ('augmenter', ImageAugmenter()),
    ('flattener', ImageFlattener()),
])

# Cell 3: Use pipeline - no changes
X_transformed = pipeline.fit_transform(image_paths, y)
```

## Benefits

1. **Cleaner Notebooks**: No need to define transformers in notebooks
2. **Reusability**: Use transformers across multiple notebooks and scripts
3. **Maintainability**: Single source of truth for transformer implementations
4. **Testability**: Transformers can be unit tested independently
5. **Separation of Concerns**: Backend (transformers) separated from frontend (notebooks)

## No Breaking Changes

The transformer APIs remain exactly the same, so existing code using the transformers will work without modifications - you just need to update the import statements!

## Documentation

For more details on each transformer, see:
- `src/features/README.md` - Module overview
- Source files in `src/features/` - Inline documentation
