"""
Example script demonstrating how to use the advanced_eda_template notebook.

This script shows the key concepts and workflow of the advanced EDA template
without requiring actual data.

Author: Data Pipeline Team
Date: November 2025
"""

# =============================================================================
# EXAMPLE 1: Loading the Configuration
# =============================================================================

def example_configuration():
    """
    Example of how configuration is loaded in the notebook.
    The CELL_CONFIG_STANDALONE cell handles all of this automatically.
    """
    from pathlib import Path
    from src.utils.config import build_config
    
    # Detect environment (this happens automatically in the notebook)
    import os
    try:
        import google.colab
        ENV = "colab"
    except ImportError:
        is_wsl = os.path.exists('/proc/version') and 'microsoft' in open('/proc/version').read().lower()
        ENV = "wsl" if is_wsl else "local"
    
    print(f"Environment detected: {ENV}")
    
    # Build configuration
    project_root = Path(__file__).parent.parent
    config = build_config(project_root, ENV)
    
    print(f"Data directory: {config.data_dir}")
    print(f"Classes: {config.classes}")
    print(f"Image size: {config.img_size}")
    
    return config


# =============================================================================
# EXAMPLE 2: Loading and Preprocessing Data
# =============================================================================

def example_data_loading(config):
    """
    Example of how data is loaded and preprocessed in the notebook.
    """
    from src.notebooks import load_dataset, create_preprocessing_pipeline
    
    # Load dataset with images and masks
    image_paths, mask_paths, labels, labels_int = load_dataset(
        data_dir=config.data_dir,
        categories=config.classes,
        n_images_per_class=config.memory['max_images_per_class'],
        load_masks=True,  # Important: also load segmentation masks
        verbose=True
    )
    
    # Create preprocessing pipeline
    pipeline_img = create_preprocessing_pipeline(
        img_size=config.img_size,
        color_mode='L',  # Grayscale for radiographs
        mask_paths=mask_paths if len(mask_paths) > 0 else None,
        verbose=True
    )
    
    # Preprocess images
    images = pipeline_img.fit_transform(image_paths)
    images = images.astype('float32') / 255.0  # Normalize to [0, 1]
    
    print(f"Loaded {len(images)} images")
    print(f"Image shape: {images.shape}")
    
    return images, labels_int


# =============================================================================
# EXAMPLE 3: PCA Analysis
# =============================================================================

def example_pca_analysis(images, labels_int, config):
    """
    Example of PCA analysis performed in the notebook.
    """
    from sklearn.decomposition import PCA
    import numpy as np
    
    # Flatten images for PCA
    n_samples = images.shape[0]
    images_flat = images.reshape(n_samples, -1)
    
    print(f"Flattened shape: {images_flat.shape}")
    
    # Apply PCA
    n_components = config.transformers['pca']['n_components']
    pca = PCA(n_components=n_components, random_state=config.training['random_seed'])
    images_pca = pca.fit_transform(images_flat)
    
    print(f"PCA shape: {images_pca.shape}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Top 10 components variance: {pca.explained_variance_ratio_[:10].sum():.4f}")
    
    return images_pca, pca


# =============================================================================
# EXAMPLE 4: Clustering Analysis
# =============================================================================

def example_clustering(images_pca, labels_int):
    """
    Example of clustering analysis performed in the notebook.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    # K-Means clustering
    n_clusters = len(set(labels_int))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(images_pca)
    
    # Calculate metrics
    sil_score = silhouette_score(images_pca, cluster_labels)
    db_score = davies_bouldin_score(images_pca, cluster_labels)
    
    print(f"K-Means (k={n_clusters}):")
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Davies-Bouldin Index: {db_score:.4f}")
    
    return cluster_labels


# =============================================================================
# EXAMPLE 5: Statistical Tests
# =============================================================================

def example_statistical_tests(images, labels_int):
    """
    Example of statistical tests performed in the notebook.
    """
    from scipy.stats import f_oneway, kruskal
    import numpy as np
    
    # Calculate mean intensity for each image
    mean_intensities = np.mean(images, axis=(1, 2, 3))
    
    # Group by class
    n_classes = len(set(labels_int))
    groups = [mean_intensities[labels_int == i] for i in range(n_classes)]
    
    # ANOVA test
    f_stat, p_value_anova = f_oneway(*groups)
    print(f"ANOVA Test:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value_anova:.6f}")
    print(f"  Significant: {'Yes' if p_value_anova < 0.05 else 'No'}")
    
    # Kruskal-Wallis test (non-parametric)
    h_stat, p_value_kruskal = kruskal(*groups)
    print(f"\nKruskal-Wallis Test:")
    print(f"  H-statistic: {h_stat:.4f}")
    print(f"  p-value: {p_value_kruskal:.6f}")
    print(f"  Significant: {'Yes' if p_value_kruskal < 0.05 else 'No'}")


# =============================================================================
# EXAMPLE 6: Feature Extraction
# =============================================================================

def example_feature_extraction(images, labels_int):
    """
    Example of textural feature extraction performed in the notebook.
    """
    from skimage.feature import graycomatrix, graycoprops
    import numpy as np
    
    # Extract features from a sample image
    sample_idx = 0
    img = images[sample_idx]
    img_uint8 = (img * 255).astype('uint8').squeeze()
    
    # Calculate GLCM
    glcm = graycomatrix(
        img_uint8,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )
    
    # Calculate GLCM properties
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    
    print(f"Textural Features (sample image):")
    print(f"  Contrast: {contrast:.4f}")
    print(f"  Homogeneity: {homogeneity:.4f}")
    print(f"  Energy: {energy:.4f}")
    print(f"  Correlation: {correlation:.4f}")


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """
    Main workflow demonstrating the advanced EDA template usage.
    
    Note: This is a conceptual example. The actual notebook runs all these
    steps with full visualizations and HTML report generation.
    """
    print("=" * 70)
    print("ADVANCED EDA TEMPLATE - USAGE EXAMPLE")
    print("=" * 70)
    
    print("\n📋 This script demonstrates the key concepts used in the")
    print("   advanced_eda_template.ipynb notebook.")
    print("\n⚠️  Note: This requires actual data to run. The notebook")
    print("   includes automatic data loading and preprocessing.")
    
    print("\n" + "=" * 70)
    print("WORKFLOW STEPS")
    print("=" * 70)
    
    steps = [
        "1. 📥 Load configuration (CELL_CONFIG_STANDALONE)",
        "2. 📊 Load and preprocess images and masks",
        "3. 🔬 Apply PCA for dimensionality reduction",
        "4. 🌐 Apply t-SNE for non-linear visualization",
        "5. 🎯 Perform clustering (K-Means, DBSCAN, Hierarchical)",
        "6. 📈 Run statistical tests (ANOVA, Kruskal-Wallis)",
        "7. 🎨 Extract textural features (GLCM)",
        "8. 📄 Generate HTML report automatically"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print("\n" + "=" * 70)
    print("KEY FEATURES")
    print("=" * 70)
    
    features = [
        "✅ Automatic environment detection (Colab, WSL, Local)",
        "✅ Comprehensive statistical analysis",
        "✅ Advanced dimensionality reduction (PCA, t-SNE)",
        "✅ Multiple clustering algorithms with quality metrics",
        "✅ Statistical hypothesis testing",
        "✅ Textural feature extraction (GLCM)",
        "✅ Professional HTML report generation",
        "✅ JSON export for programmatic use"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print("\n" + "=" * 70)
    print("TO USE THE NOTEBOOK")
    print("=" * 70)
    print("\n1. Open: notebooks/advanced_eda_template.ipynb")
    print("2. Run all cells (Cell > Run All)")
    print("3. Wait for analysis to complete")
    print("4. Check results/eda_report/ for HTML report")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
