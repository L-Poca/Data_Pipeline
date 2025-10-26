#!/usr/bin/env python3
"""
Example Usage of Data Pipeline Package

This script demonstrates how to use the data-pipeline package after installation.

Installation:
    pip install -e .

Usage:
    python examples/basic_usage.py
"""

def example_basic_import():
    """Example: Basic import of the package."""
    print("="*60)
    print("Example 1: Basic Package Import")
    print("="*60)
    
    import src
    print(f"Package version: {src.__version__}")
    print(f"Package author: {src.__author__}")
    print()


def example_import_transformers():
    """Example: Import specific transformers."""
    print("="*60)
    print("Example 2: Import Specific Transformers")
    print("="*60)
    
    try:
        from src.features import (
            ImageLoader,
            ImageResizer,
            ImageNormalizer,
            ImageFlattener,
        )
        
        print("✅ Successfully imported transformers:")
        print("   - ImageLoader")
        print("   - ImageResizer")
        print("   - ImageNormalizer")
        print("   - ImageFlattener")
        print()
        
    except ImportError as e:
        print(f"⚠️  Import failed (this is expected if dependencies are not installed): {e}")
        print("   To use the transformers, install dependencies first:")
        print("   pip install -r requirements.txt")
        print()


def example_check_available_transformers():
    """Example: Check all available transformers."""
    print("="*60)
    print("Example 3: List All Available Transformers")
    print("="*60)
    
    # List of available transformers from the package
    transformers = [
        "ImageLoader",
        "ImageResizer",
        "ImageNormalizer",
        "ImageMasker",
        "ImageFlattener",
        "ImageBinarizer",
        "ImageAugmenter",
        "ImageRandomCropper",
        "ImageHistogram",
        "ImagePCA",
        "ImageStandardScaler",
        "VisualizeTransformer",
        "SaveTransformer",
    ]
    
    print(f"Total transformers available: {len(transformers)}")
    print("\nTransformers list:")
    for transformer in transformers:
        print(f"   - {transformer}")
    print()


def example_pipeline_usage():
    """Example: How to use transformers in a pipeline (pseudo-code)."""
    print("="*60)
    print("Example 4: Pipeline Usage (Pseudo-code)")
    print("="*60)
    
    example_code = """
# Import transformers
from src.features import (
    ImageLoader,
    ImageResizer,
    ImageNormalizer,
    ImageFlattener,
)

# Optional: Use with scikit-learn pipelines (requires scikit-learn)
# from sklearn.pipeline import Pipeline

# Create a pipeline (if scikit-learn is installed)
# pipeline = Pipeline([
#     ('loader', ImageLoader(img_size=(128, 128))),
#     ('resizer', ImageResizer(img_size=(256, 256))),
#     ('normalizer', ImageNormalizer()),
#     ('flattener', ImageFlattener()),
# ])

# Or use transformers directly without scikit-learn
loader = ImageLoader(img_size=(128, 128))
resizer = ImageResizer(img_size=(256, 256))
normalizer = ImageNormalizer()
flattener = ImageFlattener()

# Use the transformers
# image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
# images = loader.fit_transform(image_paths)
# images = resizer.fit_transform(images)
# images = normalizer.fit_transform(images)
# processed_images = flattener.fit_transform(images)
"""
    
    print(example_code)
    print()


if __name__ == "__main__":
    print("\n🔧 Data Pipeline Package - Usage Examples\n")
    
    example_basic_import()
    example_import_transformers()
    example_check_available_transformers()
    example_pipeline_usage()
    
    print("="*60)
    print("For more information, see INSTALLATION.md")
    print("="*60)
