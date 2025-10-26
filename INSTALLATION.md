# Installation Guide

## Making the Package Installable

This project is now set up as an installable Python package. You can install it in development/editable mode using pip.

## Installation

### Development Mode (Editable Install)

To install the package in editable mode (recommended for development):

```bash
pip install -e .
```

This will install the package while keeping the source code in place, so any changes you make to the code will be immediately reflected without needing to reinstall.

### Standard Installation

For a standard installation:

```bash
pip install .
```

### Installing with Dependencies

If you need to install the package along with all its dependencies:

```bash
pip install -e .
```

The dependencies are read from `requirements.txt`.

### Installing Without Dependencies

If you already have the dependencies installed or want to manage them separately:

```bash
pip install --no-deps -e .
```

## Verifying Installation

After installation, you can verify that the package is correctly installed:

```bash
python test_package_install.py
```

Or test imports manually:

```python
import src
print(src.__version__)

from src.features import ImageLoader, ImageResizer
```

## Package Structure

The package is organized as follows:

- `src/` - Main package directory
  - `features/` - Feature engineering modules
    - `Pipelines/` - Pipeline components
      - `Transformateurs/` - Image transformation classes
        - `ImageLoader` - Load images from file paths
        - `ImageResizer` - Resize images
        - `ImageNormalizer` - Normalize image pixel values
        - `ImageMasker` - Apply masking to images
        - `ImageFlattener` - Flatten images
        - `ImageBinarizer` - Binarize images
        - `ImageAugmenter` - Apply data augmentation
        - `ImageRandomCropper` - Random crop augmentation
        - `ImageHistogram` - Extract histogram features
        - `ImagePCA` - Apply PCA transformation
        - `ImageStandardScaler` - Standardize features
        - `VisualizeTransformer` - Visualize transformations
        - `SaveTransformer` - Save processed images

## Uninstalling

To uninstall the package:

```bash
pip uninstall data-pipeline
```

## Development

When working on the package in development mode (`pip install -e .`), your changes to the source code will be immediately available without needing to reinstall the package.

## Troubleshooting

### Network Issues

If you encounter network timeout issues when installing dependencies, you can:

1. Install without build isolation:
   ```bash
   pip install --no-build-isolation -e .
   ```

2. Or install dependencies separately first:
   ```bash
   pip install -r requirements.txt
   pip install --no-deps -e .
   ```

### Import Errors

If you get import errors after installation, make sure you're running Python from a directory that's not the project root, or the package imports might conflict with local modules.
