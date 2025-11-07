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
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Import from interpretability module
from src.interpretability.gradcam import GradCAM

# Configure logger
logger = logging.getLogger(__name__)


# =============================================================================
# SETUP
# =============================================================================


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


def select_sample_images(
    X_data: np.ndarray,
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
        X_data: Image data
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        n_samples: Number of samples to select per class
        strategy: Selection strategy
        random_state: Random seed

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
        n_total = min(n_samples * len(class_names), len(X_data))
        indices = np.random.choice(len(X_data), size=n_total, replace=False).tolist()

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


def run_gradcam_analysis(
    gradcam: GradCAM,
    X_data: np.ndarray,
    indices: List[int],
    descriptions: List[str],
    class_names: List[str],
    y_pred_probs: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 4),
    save_dir: Optional[Path] = None,
) -> None:
    """
    Run Grad-CAM analysis on selected samples.

    Args:
        gradcam: GradCAM object
        X_data: Image data
        indices: Indices of samples to analyze
        descriptions: Description for each sample
        class_names: List of class names
        y_pred_probs: Predicted probabilities (optional)
        figsize: Figure size per sample
        save_dir: Directory to save figures (optional)
    """
    print("=" * 70)
    print(f"ANALYSE GRAD-CAM - {len(indices)} ÉCHANTILLONS")
    print("=" * 70)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n💾 Sauvegarde dans: {save_dir}")

    for i, (idx, desc) in enumerate(zip(indices, descriptions)):
        print(f"\n[{i + 1}/{len(indices)}] {desc}")

        # Get image
        img = X_data[idx]

        # Compute Grad-CAM heatmap
        heatmap = gradcam.compute_heatmap(img, class_idx=None)  # Use predicted class

        # Import visualization function
        from src.interpretability.gradcam import visualize_gradcam

        # Get predicted class info if available
        class_name = ""
        confidence = None
        if y_pred_probs is not None:
            pred_idx = np.argmax(y_pred_probs[idx])
            class_name = class_names[pred_idx]
            confidence = y_pred_probs[idx][pred_idx]

        # Visualize
        fig = visualize_gradcam(
            img,
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
