"""
Interpretability utilities for Jupyter notebooks.

This module provides functions for:
- Grad-CAM visualization
- LIME explanations
- SHAP values
- Sample selection for analysis

Author: Data Pipeline Team
Date: November 2025
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import warnings

import numpy as np
import matplotlib.pyplot as plt
import keras

# Import from interpretability module
from src.interpretability.gradcam import GradCAM, visualize_gradcam

# Try to import LIME and SHAP
try:
    from src.interpretability.lime_explainer import LIMEImageExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME non disponible. Installez avec: pip install lime", ImportWarning)

try:
    from src.interpretability.shap_explainer import SHAPExplainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP non disponible. Installez avec: pip install shap", ImportWarning)

# Import preprocessing functions
from keras.applications.inception_v3 import preprocess_input as inception_preprocess
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# Configure logger
logger = logging.getLogger(__name__)


# =============================================================================
# SETUP
# =============================================================================


def get_preprocessing_function(model_name: str) -> callable:
    """
    Get the appropriate preprocessing function for a transfer learning model.
    
    Args:
        model_name: Name of the base model ('InceptionV3', 'VGG16', 'ResNet50', 'EfficientNetB0')
    
    Returns:
        Preprocessing function
    """
    preprocessing_map = {
        'InceptionV3': inception_preprocess,
        'VGG16': vgg16_preprocess,
        'ResNet50': resnet_preprocess,
        'EfficientNetB0': efficientnet_preprocess,
    }
    
    if model_name not in preprocessing_map:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported: {list(preprocessing_map.keys())}"
        )
    
    return preprocessing_map[model_name]


def setup_interpretability(model: keras.Model, verbose: bool = True) -> GradCAM:
    """
    Initialize Grad-CAM for model interpretability.

    Automatically finds the last convolutional layer for Grad-CAM visualization.

    Args:
        model: Trained Keras model
        verbose: Print setup information

    Returns:
        GradCAM object ready for visualization
    """
    if verbose:
        print("=" * 70)
        print("SETUP INTERPRÉTABILITÉ - GRAD-CAM")
        print("=" * 70)

    # Find last convolutional layer
    conv_layer_name = None
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            conv_layer_name = layer.name
            break

    if conv_layer_name is None:
        # For transfer learning models, need to go into base model
        for layer in reversed(model.layers):
            if hasattr(layer, "layers"):  # Sequential or Functional model
                for sublayer in reversed(layer.layers):
                    if "conv" in sublayer.name.lower():
                        conv_layer_name = sublayer.name
                        break
                if conv_layer_name:
                    break

    if conv_layer_name is None:
        raise ValueError("Aucune couche convolutionnelle trouvée dans le modèle!")

    if verbose:
        print("\n✅ Grad-CAM configuré")
        print(f"   Couche convolutionnelle: {conv_layer_name}")
        print(f"   Modèle: {model.name}")

    # Initialize Grad-CAM
    gradcam = GradCAM(model, conv_layer_name)

    return gradcam


# =============================================================================
# SAMPLE SELECTION
# =============================================================================


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
def select_sample_images(
    x_data: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    n_samples: int = 2,
    strategy: str = "correct",
    random_seed: int = 42,
) -> Tuple[List[int], List[str]]:
    """
    Select sample images for Grad-CAM analysis.

    Strategies:
        - 'correct': Correctly classified images
        - 'incorrect': Misclassified images
        - 'random': Random selection
        - 'one_per_class': One sample per class (correctly classified)

    Args:
        x_data: Image data
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        n_samples: Number of samples to select per class
        strategy: Selection strategy
        random_seed: Random seed

    Returns:
        Tuple of (indices, descriptions)
    """
    print("=" * 70)
    print(f"SÉLECTION D'ÉCHANTILLONS - {strategy.upper()}")
    print("=" * 70)

    np.random.seed(random_seed)

    indices = []
    descriptions = []

    if strategy == "correct":
        # Select correctly classified samples
        for class_idx, class_name in enumerate(class_names):
            mask = (y_true == class_idx) & (y_pred == class_idx)
            class_indices = np.where(mask)[0]

            if len(class_indices) > 0:
                selected = np.random.choice(
                    class_indices,
                    size=min(n_samples, len(class_indices)),
                    replace=False,
                )
                indices.extend(selected)
                descriptions.extend([f"{class_name} (correct)"] * len(selected))

        print(f"\n✅ {len(indices)} échantillons correctement classifiés sélectionnés")

    elif strategy == "incorrect":
        # Select misclassified samples
        for class_idx, class_name in enumerate(class_names):
            mask = (y_true == class_idx) & (y_pred != class_idx)
            class_indices = np.where(mask)[0]

            if len(class_indices) > 0:
                selected = np.random.choice(
                    class_indices,
                    size=min(n_samples, len(class_indices)),
                    replace=False,
                )
                indices.extend(selected)

                for idx in selected:
                    pred_class = class_names[y_pred[idx]]
                    descriptions.append(f"{class_name} → {pred_class} (incorrect)")

        print(f"\n✅ {len(indices)} échantillons mal classifiés sélectionnés")

    elif strategy == "one_per_class":
        # Select one correct sample per class
        for class_idx, class_name in enumerate(class_names):
            mask = (y_true == class_idx) & (y_pred == class_idx)
            class_indices = np.where(mask)[0]

            if len(class_indices) > 0:
                selected = np.random.choice(class_indices, size=1)
                indices.extend(selected)
                descriptions.append(f"{class_name} (correct)")

        print(f"\n✅ {len(indices)} échantillons sélectionnés (1 par classe)")

    elif strategy == "random":
        # Random selection
        n_total = min(n_samples * len(class_names), len(x_data))
        indices = np.random.choice(len(x_data), size=n_total, replace=False).tolist()

        for idx in indices:
            true_class = class_names[y_true[idx]]
            pred_class = class_names[y_pred[idx]]
            correct = "✓" if y_true[idx] == y_pred[idx] else "✗"
            descriptions.append(f"{true_class} → {pred_class} {correct}")

        print(f"\n✅ {len(indices)} échantillons aléatoires sélectionnés")

    else:
        raise ValueError(f"Stratégie inconnue: {strategy}")

    # Print summary
    print("\n📊 Résumé:")
    print(f"   Total: {len(indices)} échantillons")
    for desc in set(descriptions):
        count = descriptions.count(desc)
        print(f"   • {desc}: {count}")

    return indices, descriptions


# =============================================================================
# GRAD-CAM ANALYSIS
# =============================================================================


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def run_gradcam_analysis(
    gradcam: GradCAM,
    x_data: np.ndarray,
    indices: List[int],
    descriptions: List[str],
    class_names: List[str],
    y_pred_probs: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None,
    preprocess_fn: Optional[callable] = None,
) -> None:
    """
    Run Grad-CAM analysis on selected samples.

    Args:
        gradcam: GradCAM object
        x_data: Image data (raw images, not preprocessed)
        indices: Indices of samples to analyze
        descriptions: Description for each sample
        class_names: List of class names
        y_pred_probs: Predicted probabilities (optional)
        save_dir: Directory to save figures (optional)
        preprocess_fn: Preprocessing function to apply (e.g., InceptionV3 preprocessing)
    """
    print("=" * 70)
    print(f"ANALYSE GRAD-CAM - {len(indices)} ÉCHANTILLONS")
    print("=" * 70)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n💾 Sauvegarde dans: {save_dir}")
    
    if preprocess_fn is not None:
        print("⚠️  Preprocessing appliqué aux images pour Grad-CAM")

    for i, (idx, desc) in enumerate(zip(indices, descriptions)):
        print(f"\n[{i + 1}/{len(indices)}] {desc}")

        # Get image (raw, pour visualisation)
        img_raw = x_data[idx].copy()
        
        # Apply preprocessing if provided (pour le modèle)
        if preprocess_fn is not None:
            img_preprocessed = preprocess_fn(x_data[idx].copy())
        else:
            img_preprocessed = img_raw

        # Compute Grad-CAM heatmap (avec image preprocessée)
        heatmap = gradcam.compute_heatmap(img_preprocessed, class_idx=None)  # Use predicted class

        # Get predicted class info if available
        class_name = ""
        confidence = None
        if y_pred_probs is not None:
            pred_idx = np.argmax(y_pred_probs[idx])
            class_name = class_names[pred_idx]
            confidence = y_pred_probs[idx][pred_idx]

        # Visualize (avec image RAW pour affichage correct)
        # Normaliser l'image raw pour affichage (0-1)
        img_display = img_raw.astype('float32')
        if img_display.max() > 1:
            img_display = img_display / 255.0
        
        fig = visualize_gradcam(
            img_display,
            heatmap,
            class_name=class_name,
            confidence=confidence,
            alpha=0.4,
            colormap="jet",
        )

        # Add description as super title (override the default title)
        fig.suptitle(desc, fontsize=14, fontweight="bold", y=1.02)

        # Save if requested
        if save_dir:
            save_path = save_dir / f"gradcam_{i + 1:02d}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"   💾 Sauvegardé: {save_path.name}")

        plt.show()

    print("\n✅ Analyse Grad-CAM terminée!")


# =============================================================================
# LIME ANALYSIS
# =============================================================================


def setup_lime_explainer(
    model: keras.Model,
    segmentation_method: str = 'quickshift',
    num_samples: int = 1000,
    batch_size: int = 32,
    verbose: bool = True
) -> Optional[LIMEImageExplainer]:
    """
    Initialize LIME explainer for model interpretability.

    Args:
        model: Trained Keras model
        segmentation_method: Method for image segmentation ('quickshift', 'felzenszwalb', 'slic')
        num_samples: Number of perturbed samples for LIME
        batch_size: Batch size for predictions
        verbose: Print setup information

    Returns:
        LIMEImageExplainer object or None if LIME not available
    """
    if not LIME_AVAILABLE:
        print("⚠️  LIME non disponible. Installez avec: pip install lime")
        return None

    if verbose:
        print("=" * 70)
        print("SETUP INTERPRÉTABILITÉ - LIME")
        print("=" * 70)

    # Initialize LIME
    lime_explainer = LIMEImageExplainer(
        predict_fn=model.predict,
        segmentation_method=segmentation_method,
        num_samples=num_samples,
        batch_size=batch_size
    )

    if verbose:
        print("\n✅ LIME configuré")
        print(f"   Méthode de segmentation: {segmentation_method}")
        print(f"   Nombre d'échantillons: {num_samples}")
        print(f"   Batch size: {batch_size}")

    return lime_explainer


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def run_lime_analysis(
    lime_explainer: LIMEImageExplainer,
    x_data: np.ndarray,
    indices: List[int],
    descriptions: List[str],
    class_names: List[str],
    y_pred: Optional[np.ndarray] = None,
    num_features: int = 5,
    save_dir: Optional[Path] = None,
    show_boundaries: bool = True,
) -> None:
    """
    Run LIME analysis on selected samples.

    Args:
        lime_explainer: LIME explainer object
        x_data: Image data (normalized 0-1)
        indices: Indices of samples to analyze
        descriptions: Description for each sample
        class_names: List of class names
        y_pred: Predicted class indices (optional)
        num_features: Number of superpixels to highlight
        save_dir: Directory to save figures (optional)
        show_boundaries: Show superpixel boundaries
    """
    if lime_explainer is None:
        print("⚠️  LIME non disponible")
        return

    print("=" * 70)
    print(f"ANALYSE LIME - {len(indices)} ÉCHANTILLONS")
    print("=" * 70)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n💾 Sauvegarde dans: {save_dir}")

    for i, (idx, desc) in enumerate(zip(indices, descriptions)):
        print(f"\n[{i + 1}/{len(indices)}] {desc}")

        # Get image
        img = x_data[idx].copy()
        
        # Normalize if needed
        if img.max() > 1:
            img = img / 255.0

        # Get predicted class
        if y_pred is not None:
            pred_class = y_pred[idx]
        else:
            pred_probs = lime_explainer.predict_fn(img[np.newaxis, ...])
            pred_class = np.argmax(pred_probs[0])

        # Generate LIME explanation
        print(f"   Génération de l'explication LIME (classe: {class_names[pred_class]})...")
        explanation = lime_explainer.explain_instance(
            img,
            top_labels=1,
            num_features=num_features,
            random_seed=42 + i
        )

        # Visualize
        if show_boundaries:
            fig = lime_explainer.visualize_explanation_boundaries(
                img,
                explanation,
                label=pred_class,
                num_features=num_features,
                figsize=(12, 5)
            )
        else:
            fig = lime_explainer.visualize_explanation(
                img,
                explanation,
                label=pred_class,
                num_features=num_features,
                positive_only=True,
                hide_rest=False,
                figsize=(15, 5)
            )

        # Add description as super title
        fig.suptitle(f"LIME - {desc}", fontsize=14, fontweight="bold", y=1.02)

        # Save if requested
        if save_dir:
            save_path = save_dir / f"lime_{i + 1:02d}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"   💾 Sauvegardé: {save_path.name}")

        plt.show()

        # Optionally show feature contributions
        fig_contrib = lime_explainer.visualize_top_features(
            explanation,
            label=pred_class,
            num_features=num_features,
            figsize=(10, 6)
        )
        
        if save_dir:
            save_path_contrib = save_dir / f"lime_contrib_{i + 1:02d}.png"
            plt.savefig(save_path_contrib, dpi=300, bbox_inches="tight")
            print(f"   💾 Contributions sauvegardées: {save_path_contrib.name}")
        
        plt.show()

    print("\n✅ Analyse LIME terminée!")


def compare_lime_segmentation(
    lime_explainer: LIMEImageExplainer,
    image: np.ndarray,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Compare different segmentation methods for LIME.

    Args:
        lime_explainer: LIME explainer (any config, will be overridden)
        image: Image to segment
        save_dir: Directory to save figure
    """
    if lime_explainer is None:
        print("⚠️  LIME non disponible")
        return

    print("=" * 70)
    print("COMPARAISON DES MÉTHODES DE SEGMENTATION LIME")
    print("=" * 70)

    # Normalize image
    if image.max() > 1:
        image = image / 255.0

    fig = lime_explainer.compare_segmentation_methods(
        image,
        methods=['quickshift', 'felzenszwalb', 'slic'],
        figsize=(16, 5)
    )

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "lime_segmentation_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"   💾 Sauvegardé: {save_path}")

    plt.show()

    print("\n✅ Comparaison terminée!")


# =============================================================================
# SHAP ANALYSIS
# =============================================================================


def setup_shap_explainer(
    model: keras.Model,
    background_data: np.ndarray,
    max_background_samples: int = 100,
    verbose: bool = True
) -> Optional[SHAPExplainer]:
    """
    Initialize SHAP explainer for model interpretability.

    Args:
        model: Trained Keras model
        background_data: Reference data for SHAP (training subset)
        max_background_samples: Maximum number of background samples (SHAP can be slow)
        verbose: Print setup information

    Returns:
        SHAPExplainer object or None if SHAP not available
    """
    if not SHAP_AVAILABLE:
        print("⚠️  SHAP non disponible. Installez avec: pip install shap")
        return None

    if verbose:
        print("=" * 70)
        print("SETUP INTERPRÉTABILITÉ - SHAP")
        print("=" * 70)

    # Limit background data size
    if len(background_data) > max_background_samples:
        print(f"   Limitation du background à {max_background_samples} échantillons...")
        background_subset = background_data[:max_background_samples]
    else:
        background_subset = background_data

    # Initialize SHAP
    shap_explainer = SHAPExplainer(
        model=model,
        background_data=background_subset
    )

    if verbose:
        print("\n✅ SHAP configuré")
        print(f"   Background samples: {len(background_subset)}")
        print(f"   ⚠️  Note: SHAP peut être lent pour les CNN")

    return shap_explainer


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def run_shap_analysis(
    shap_explainer: SHAPExplainer,
    x_data: np.ndarray,
    indices: List[int],
    descriptions: List[str],
    class_names: List[str],
    y_pred: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Run SHAP analysis on selected samples.

    Args:
        shap_explainer: SHAP explainer object
        x_data: Image data
        indices: Indices of samples to analyze
        descriptions: Description for each sample
        class_names: List of class names
        y_pred: Predicted class indices (optional)
        save_dir: Directory to save figures (optional)
    """
    if shap_explainer is None:
        print("⚠️  SHAP non disponible")
        return

    print("=" * 70)
    print(f"ANALYSE SHAP - {len(indices)} ÉCHANTILLONS")
    print("=" * 70)
    print("⚠️  Calcul des valeurs SHAP en cours (cela peut prendre du temps)...")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n💾 Sauvegarde dans: {save_dir}")

    # Get images to analyze
    images_to_analyze = x_data[indices]

    # Calculate SHAP values for all images at once
    print(f"\n   Calcul des valeurs SHAP pour {len(indices)} images...")
    shap_values = shap_explainer.explain(images_to_analyze)

    # Visualize each image
    for i, (idx, desc) in enumerate(zip(indices, descriptions)):
        print(f"\n[{i + 1}/{len(indices)}] {desc}")

        img = x_data[idx].copy()

        # Get predicted class
        if y_pred is not None:
            pred_class = y_pred[idx]
        else:
            pred_probs = shap_explainer.model.predict(img[np.newaxis, ...], verbose=0)
            pred_class = np.argmax(pred_probs[0])

        class_name = class_names[pred_class]

        # Visualize SHAP values
        fig = shap_explainer.visualize_image_plot(
            img,
            shap_values[i],
            class_idx=pred_class,
            class_name=class_name,
            figsize=(12, 4)
        )

        # Add description
        fig.suptitle(f"SHAP - {desc}", fontsize=14, fontweight="bold", y=1.02)

        # Save if requested
        if save_dir:
            save_path = save_dir / f"shap_{i + 1:02d}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"   💾 Sauvegardé: {save_path.name}")

        plt.show()

        # Also create heatmap overlay
        fig_heatmap = shap_explainer.visualize_heatmap(
            img,
            shap_values[i],
            class_idx=pred_class,
            alpha=0.4,
            colormap='jet',
            figsize=(12, 4)
        )

        if save_dir:
            save_path_heatmap = save_dir / f"shap_heatmap_{i + 1:02d}.png"
            plt.savefig(save_path_heatmap, dpi=300, bbox_inches="tight")
            print(f"   💾 Heatmap sauvegardé: {save_path_heatmap.name}")

        plt.show()

    print("\n✅ Analyse SHAP terminée!")


def compare_shap_classes(
    shap_explainer: SHAPExplainer,
    image: np.ndarray,
    class_names: List[str],
    save_dir: Optional[Path] = None,
) -> None:
    """
    Compare SHAP values across all classes for a single image.

    Args:
        shap_explainer: SHAP explainer object
        image: Image to analyze
        class_names: List of class names
        save_dir: Directory to save figure
    """
    if shap_explainer is None:
        print("⚠️  SHAP non disponible")
        return

    print("=" * 70)
    print("COMPARAISON SHAP INTER-CLASSES")
    print("=" * 70)

    # Calculate SHAP values
    print("   Calcul des valeurs SHAP...")
    shap_values = shap_explainer.explain(image[np.newaxis, ...])

    # Visualize comparison
    fig = shap_explainer.compare_classes(
        image,
        shap_values[0],
        class_names,
        figsize=(16, 4)
    )

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "shap_class_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"   💾 Sauvegardé: {save_path}")

    plt.show()

    print("\n✅ Comparaison terminée!")


# =============================================================================
# COMBINED ANALYSIS
# =============================================================================


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def run_full_interpretability_analysis(
    model: keras.Model,
    x_data: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    background_data: Optional[np.ndarray] = None,
    n_samples: int = 2,
    strategy: str = "one_per_class",
    save_dir: Optional[Path] = None,
    use_gradcam: bool = True,
    use_lime: bool = True,
    use_shap: bool = False,  # SHAP disabled by default (slow)
    preprocess_fn: Optional[Callable] = None,
) -> None:
    """
    Run complete interpretability analysis with Grad-CAM, LIME, and SHAP.

    Args:
        model: Trained Keras model
        x_data: Image data
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        background_data: Background data for SHAP (training subset)
        n_samples: Number of samples per class
        strategy: Selection strategy ('correct', 'incorrect', 'one_per_class', 'random')
        save_dir: Directory to save all figures
        use_gradcam: Run Grad-CAM analysis
        use_lime: Run LIME analysis
        use_shap: Run SHAP analysis (WARNING: slow)
        preprocess_fn: Preprocessing function for Grad-CAM
    """
    print("=" * 70)
    print("ANALYSE D'INTERPRÉTABILITÉ COMPLÈTE")
    print("=" * 70)
    print(f"\nMéthodes activées:")
    print(f"   • Grad-CAM: {'✓' if use_gradcam else '✗'}")
    print(f"   • LIME: {'✓' if use_lime else '✗'}")
    print(f"   • SHAP: {'✓' if use_shap else '✗'}")

    # Create save directories
    gradcam_dir = save_dir / "gradcam" if save_dir else None
    lime_dir = save_dir / "lime" if save_dir else None
    shap_dir = save_dir / "shap" if save_dir else None

    # Select samples
    indices, descriptions = select_sample_images(
        x_data, y_true, y_pred, class_names,
        n_samples=n_samples,
        strategy=strategy
    )
    
    # Extract selected data and predictions
    x_selected = x_data[indices]
    y_pred_selected = y_pred[indices]

    # Get predicted probabilities
    print("\nCalcul des probabilités de prédiction...")
    y_pred_probs = model.predict(x_selected, verbose=0)

    # 1. Grad-CAM Analysis
    if use_gradcam:
        try:
            print("\n" + "=" * 70)
            print("1/3 - GRAD-CAM")
            print("=" * 70)
            gradcam = setup_interpretability(model, verbose=True)
            run_gradcam_analysis(
                gradcam, x_selected, list(range(len(indices))), descriptions, class_names,
                y_pred_probs=y_pred_probs,
                save_dir=gradcam_dir,
                preprocess_fn=preprocess_fn
            )
        except Exception as e:
            print(f"⚠️  Erreur Grad-CAM: {e}")

    # 2. LIME Analysis
    if use_lime and LIME_AVAILABLE:
        try:
            print("\n" + "=" * 70)
            print("2/3 - LIME")
            print("=" * 70)
            lime_explainer = setup_lime_explainer(
                model,
                segmentation_method='quickshift',
                num_samples=1000,
                verbose=True
            )
            if lime_explainer:
                run_lime_analysis(
                    lime_explainer, x_selected, list(range(len(indices))), descriptions, class_names,
                    y_pred=y_pred_selected,
                    num_features=5,
                    save_dir=lime_dir
                )
        except Exception as e:
            print(f"⚠️  Erreur LIME: {e}")
    elif use_lime:
        print("\n⚠️  LIME non disponible (pip install lime)")

    # 3. SHAP Analysis
    if use_shap and SHAP_AVAILABLE:
        try:
            print("\n" + "=" * 70)
            print("3/3 - SHAP")
            print("=" * 70)
            if background_data is None:
                print("⚠️  Background data requis pour SHAP. Utilisation d'un subset des données.")
                background_data = x_data[:50]
            
            shap_explainer = setup_shap_explainer(
                model,
                background_data,
                max_background_samples=50,
                verbose=True
            )
            if shap_explainer:
                run_shap_analysis(
                    shap_explainer, x_selected, list(range(len(indices))), descriptions, class_names,
                    y_pred=y_pred_selected,
                    save_dir=shap_dir
                )
        except Exception as e:
            print(f"⚠️  Erreur SHAP: {e}")
    elif use_shap:
        print("\n⚠️  SHAP non disponible (pip install shap)")

    print("\n" + "=" * 70)
    print("✅ ANALYSE D'INTERPRÉTABILITÉ COMPLÈTE TERMINÉE")
    print("=" * 70)
    if save_dir:
        print(f"\n📁 Tous les résultats sauvegardés dans: {save_dir}")
        print(f"   • Grad-CAM: {gradcam_dir}")
        print(f"   • LIME: {lime_dir}")
        print(f"   • SHAP: {shap_dir}")
