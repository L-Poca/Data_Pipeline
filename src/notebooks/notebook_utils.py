"""
Utility functions for Jupyter notebooks.

This module provides reusable functions for common notebook operations:
- Data loading and preprocessing
- Model building and training
- Evaluation and visualization
- Interpretability analysis

Author: Data Pipeline Team
Date: November 2025
"""

import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Configure logger
logger = logging.getLogger(__name__)


# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================

def load_dataset(
    data_dir: Path,
    categories: List[str],
    n_images_per_class: Optional[int] = None,
    load_masks: bool = False,
    verbose: bool = True
) -> Tuple[List[Path], List[Path], List[str], np.ndarray]:
    """
    Load image paths and labels from dataset directory.
    
    Args:
        data_dir: Root directory of the dataset
        categories: List of category names
        n_images_per_class: Maximum number of images per class (None = all)
        load_masks: If True, also load mask paths
        verbose: Print loading information
    
    Returns:
        Tuple of (image_paths, mask_paths, labels, labels_int)
        If load_masks=False, mask_paths will be empty list
    """
    if verbose:
        print("=" * 70)
        print("CHARGEMENT DES DONNÉES")
        print("=" * 70)
    
    image_paths = []
    mask_paths = []
    labels = []
    labels_int = []
    
    for idx, cat in enumerate(categories):
        cat_path = data_dir / cat / 'images'
        
        if not cat_path.exists():
            logger.warning(f"Category path not found: {cat_path}")
            continue
        
        imgs = sorted(list(cat_path.glob('*.png')))
        
        if n_images_per_class:
            imgs = imgs[:n_images_per_class]
        
        if load_masks:
            # Load corresponding masks
            mask_cat_path = data_dir / cat / 'masks'
            for img_path in imgs:
                mask_path = mask_cat_path / img_path.name
                if mask_path.exists():
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
                    labels.append(cat)
                    labels_int.append(idx)
        else:
            # Load only images
            image_paths.extend(imgs)
            labels.extend([cat] * len(imgs))
            labels_int.extend([idx] * len(imgs))
        
        if verbose:
            suffix = " (avec masques)" if load_masks else ""
            print(f"  {cat:20s}: {len(imgs):4d} images{suffix}")
    
    labels_int = np.array(labels_int)
    
    if verbose:
        print(f"\n  Total: {len(image_paths)} images")
        print(f"  Classes: {len(categories)}")
        print(f"  Distribution: {np.bincount(labels_int)}")
        if load_masks:
            print(f"\n✅ Chemins des masques récupérés: {len(mask_paths)}")
    
    return image_paths, mask_paths, labels, labels_int


def create_preprocessing_pipeline(
    img_size: Tuple[int, int] = (128, 128),
    color_mode: str = 'RGB',
    mask_paths: Optional[List[Path]] = None,
    verbose: bool = True
) -> Pipeline:
    """
    Create a preprocessing pipeline for images.
    
    Args:
        img_size: Target image size (width, height)
        color_mode: 'RGB' or 'L' (grayscale)
        mask_paths: Optional mask paths for ImageMasker
        verbose: Print pipeline information
    
    Returns:
        sklearn Pipeline
    """
    from src.features.Pipelines.Transformateurs.image_loaders import ImageLoader
    from src.features.Pipelines.Transformateurs.image_preprocessing import (
        ImageResizer, ImageMasker
    )
    
    if verbose:
        print("=" * 70)
        print("PREPROCESSING PIPELINE")
        print("=" * 70)
    
    steps = [
        ('load', ImageLoader(color_mode=color_mode, verbose=verbose)),
        ('resize', ImageResizer(img_size=img_size, verbose=verbose))
    ]
    
    # Add masker if mask paths provided
    if mask_paths is not None and len(mask_paths) > 0:
        steps.append(('mask', ImageMasker(mask_paths=mask_paths, verbose=verbose)))
    
    pipeline = Pipeline(steps)
    
    if verbose:
        print(f"\n✅ Pipeline créée avec {len(steps)} étapes")
    
    return pipeline


def prepare_train_val_test_split(
    images: np.ndarray,
    labels_int: np.ndarray,
    num_classes: int,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/validation/test sets with one-hot encoding.
    
    Args:
        images: Array of images
        labels_int: Integer labels
        num_classes: Number of classes
        test_size: Proportion of test set
        val_size: Proportion of validation set (from total)
        random_seed: Random seed for reproducibility
        verbose: Print split information
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat)
    """
    if verbose:
        print("=" * 70)
        print("TRAIN/VALIDATION/TEST SPLIT")
        print("=" * 70)
    
    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, labels_int,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels_int
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size_adjusted,
        random_state=random_seed,
        stratify=y_train_val
    )
    
    # One-hot encoding
    y_train_cat = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes=num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes=num_classes)
    
    if verbose:
        print(f"\nTrain set: {X_train.shape[0]} images")
        print(f"  Distribution: {np.bincount(y_train)}")
        print(f"\nValidation set: {X_val.shape[0]} images")
        print(f"  Distribution: {np.bincount(y_val)}")
        print(f"\nTest set: {X_test.shape[0]} images")
        print(f"  Distribution: {np.bincount(y_test)}")
    
    return X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat


def compute_class_weights(
    y_train: np.ndarray,
    categories: List[str],
    verbose: bool = True
) -> Dict[int, float]:
    """
    Compute balanced class weights.
    
    Args:
        y_train: Training labels (integer)
        categories: List of category names
        verbose: Print weight information
    
    Returns:
        Dictionary mapping class index to weight
    """
    if verbose:
        print("=" * 70)
        print("CLASS WEIGHTING")
        print("=" * 70)
    
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
    
    if verbose:
        print("\nPoids de classe:")
        for i, cat in enumerate(categories):
            print(f"  {cat:20s}: {class_weights[i]:.3f}")
        print("\n✅ Class weighting activé pour un apprentissage équilibré")
    
    return class_weights


def create_data_generators(
    X_train: np.ndarray,
    y_train_cat: np.ndarray,
    X_val: np.ndarray,
    y_val_cat: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test_cat: Optional[np.ndarray] = None,
    batch_size: int = 32,
    augment_train: bool = True,
    verbose: bool = True
) -> Tuple[Any, Any, Optional[Any]]:
    """
    Create Keras data generators with optional augmentation.
    
    Args:
        X_train: Training images
        y_train_cat: Training labels (one-hot)
        X_val: Validation images
        y_val_cat: Validation labels (one-hot)
        X_test: Test images (optional)
        y_test_cat: Test labels (one-hot, optional)
        batch_size: Batch size
        augment_train: Apply augmentation to training data
        verbose: Print generator information
    
    Returns:
        Tuple of (train_generator, val_generator, test_generator)
        test_generator is None if X_test not provided
    """
    if verbose:
        print("=" * 70)
        print("DATA AUGMENTATION")
        print("=" * 70)
    
    # Training generator with augmentation
    if augment_train:
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        if verbose:
            print("\n✅ Data augmentation configurée:")
            print("  • Rotation: ±10°")
            print("  • Shift: ±10%")
            print("  • Zoom: ±10%")
            print("  • Horizontal flip")
    else:
        train_datagen = ImageDataGenerator()
        if verbose:
            print("\n⚠️ Pas d'augmentation sur le training set")
    
    # Validation and test generators (no augmentation)
    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    
    if verbose:
        print("\n📊 Création des générateurs...")
    
    train_generator = train_datagen.flow(
        X_train, y_train_cat,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val_cat,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_generator = None
    if X_test is not None and y_test_cat is not None:
        test_generator = test_datagen.flow(
            X_test, y_test_cat,
            batch_size=batch_size,
            shuffle=False
        )
    
    if verbose:
        print(f"  Train: {len(train_generator)} batches de {batch_size}")
        print(f"  Val:   {len(val_generator)} batches de {batch_size}")
        if test_generator:
            print(f"  Test:  {len(test_generator)} batches de {batch_size}")
    
    return train_generator, val_generator, test_generator


# =============================================================================
# 2. MODEL BUILDING
# =============================================================================

def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 4,
    verbose: bool = True
) -> keras.Model:
    """
    Build a custom CNN architecture optimized for medical imaging.
    
    Architecture:
        - 5 convolutional blocks (32→64→128→256→512 filters)
        - Batch normalization after each Conv2D
        - Dropout for regularization
        - L2 regularization on dense layers
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        verbose: Print model information
    
    Returns:
        Compiled Keras model
    """
    if verbose:
        print("=" * 70)
        print("CUSTOM CNN ARCHITECTURE")
        print("=" * 70)
    
    model = models.Sequential(name='CustomCNN_COVID19')
    
    # Bloc 1: 32 filtres
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Bloc 2: 64 filtres
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Bloc 3: 128 filtres
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Bloc 4: 256 filtres
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Bloc 5: 512 filtres
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))
    
    # Flatten et couches denses
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    # Couche de sortie
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    if verbose:
        print("\n✅ Modèle créé")
        print(f"   Nom: {model.name}")
        print(f"   Input shape: {input_shape}")
        print(f"   Output classes: {num_classes}")
    
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    verbose: bool = True
) -> keras.Model:
    """
    Compile a Keras model with standard metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        verbose: Print compilation information
    
    Returns:
        Compiled model
    """
    if verbose:
        print("=" * 70)
        print("COMPILATION DU MODÈLE")
        print("=" * 70)
    
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(),
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    if verbose:
        print("\n✅ Modèle compilé")
        print(f"   Optimizer: {optimizer.__class__.__name__}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Loss: CategoricalCrossentropy")
        print(f"   Metrics: accuracy, auc, precision, recall")
    
    return model


def create_callbacks(
    models_dir: Path,
    monitor: str = 'val_accuracy',
    patience_early_stop: int = 15,
    patience_reduce_lr: int = 5,
    verbose: bool = True
) -> List[keras.callbacks.Callback]:
    """
    Create standard training callbacks.
    
    Args:
        models_dir: Directory to save models
        monitor: Metric to monitor
        patience_early_stop: Patience for early stopping
        patience_reduce_lr: Patience for learning rate reduction
        verbose: Print callback information
    
    Returns:
        List of Keras callbacks
    """
    if verbose:
        print("=" * 70)
        print("CALLBACKS")
        print("=" * 70)
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1 if verbose else 0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_reduce_lr,
            min_lr=1e-7,
            verbose=1 if verbose else 0
        ),
        ModelCheckpoint(
            filepath=str(models_dir / 'best_model.keras'),
            monitor=monitor,
            save_best_only=True,
            verbose=1 if verbose else 0
        )
    ]
    
    if verbose:
        print(f"\n✅ Callbacks configurés:")
        print(f"   • EarlyStopping (patience={patience_early_stop})")
        print(f"   • ReduceLROnPlateau (factor=0.5, patience={patience_reduce_lr})")
        print(f"   • ModelCheckpoint (monitor={monitor})")
    
    return callbacks


# =============================================================================
# 3. TRAINING & EVALUATION
# =============================================================================

def train_model(
    model: keras.Model,
    train_generator: Any,
    val_generator: Any,
    class_weights: Dict[int, float],
    epochs: int = 50,
    callbacks: Optional[List[keras.callbacks.Callback]] = None,
    verbose: bool = True
) -> keras.callbacks.History:
    """
    Train a Keras model.
    
    Args:
        model: Compiled Keras model
        train_generator: Training data generator
        val_generator: Validation data generator
        class_weights: Class weights dictionary
        epochs: Number of training epochs
        callbacks: List of callbacks
        verbose: Verbosity level
    
    Returns:
        Training history
    """
    if verbose:
        print("=" * 70)
        print("ENTRAÎNEMENT DU MODÈLE")
        print("=" * 70)
        print(f"\nDébut de l'entraînement: {epochs} époques")
        print(f"Class weights activé: ✅")
        print("\n" + "=" * 70 + "\n")
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=callbacks if callbacks else [],
        verbose=1 if verbose else 0
    )
    
    if verbose:
        print("\n" + "=" * 70)
        print("✅ ENTRAÎNEMENT TERMINÉ")
        print("=" * 70)
    
    return history


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test_cat: np.ndarray,
    y_test: np.ndarray,
    categories: List[str],
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model on test set and print metrics.
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test_cat: Test labels (one-hot)
        y_test: Test labels (integer)
        categories: List of category names
        verbose: Print evaluation results
    
    Returns:
        Tuple of (y_pred, y_pred_proba)
    """
    if verbose:
        print("=" * 70)
        print("ÉVALUATION SUR LE TEST SET")
        print("=" * 70)
        print("\nÉvaluation en cours...")
    
    # Evaluate
    test_loss, test_acc, test_auc, test_prec, test_rec = model.evaluate(
        X_test, y_test_cat, verbose=0
    )
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    if verbose:
        print(f"\n📊 Résultats sur le test set:")
        print(f"   Loss:      {test_loss:.4f}")
        print(f"   Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"   AUC:       {test_auc:.4f}")
        print(f"   Precision: {test_prec:.4f}")
        print(f"   Recall:    {test_rec:.4f}")
        print(f"   F1-Score:  {2*(test_prec*test_rec)/(test_prec+test_rec):.4f}")
        
        print("\n" + "=" * 70)
        print("RAPPORT DE CLASSIFICATION")
        print("=" * 70)
        print()
        print(classification_report(y_test, y_pred, target_names=categories, digits=4))
    
    return y_pred, y_pred_proba


# =============================================================================
# 4. VISUALIZATION
# =============================================================================

def plot_training_curves(
    history: keras.callbacks.History,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 12)
) -> plt.Figure:
    """
    Plot training curves (loss, accuracy, AUC, precision, recall).
    
    Args:
        history: Training history
        save_path: Path to save figure (optional)
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    print("=" * 70)
    print("VISUALISATION DES COURBES D'APPRENTISSAGE")
    print("=" * 70)
    
    hist = history.history
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Loss
    axes[0, 0].plot(hist['loss'], label='Train', linewidth=2)
    axes[0, 0].plot(hist['val_loss'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=14, weight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(hist['accuracy'], label='Train', linewidth=2)
    axes[0, 1].plot(hist['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Accuracy', fontsize=14, weight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 0].plot(hist['auc'], label='Train', linewidth=2)
    axes[1, 0].plot(hist['val_auc'], label='Validation', linewidth=2)
    axes[1, 0].set_title('AUC', fontsize=14, weight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision & Recall
    axes[1, 1].plot(hist['precision'], label='Train Precision', linewidth=2)
    axes[1, 1].plot(hist['val_precision'], label='Val Precision', linewidth=2, linestyle='--')
    axes[1, 1].plot(hist['recall'], label='Train Recall', linewidth=2)
    axes[1, 1].plot(hist['val_recall'], label='Val Recall', linewidth=2, linestyle='--')
    axes[1, 1].set_title('Precision & Recall', fontsize=14, weight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Courbes d\'Apprentissage', fontsize=16, weight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Graphique sauvegardé: {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    categories: List[str],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        categories: List of category names
        save_path: Path to save figure (optional)
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=categories, yticklabels=categories,
        cbar_kws={'label': 'Count'}
    )
    ax.set_title('Matrice de Confusion', fontsize=14, weight='bold', pad=20)
    ax.set_xlabel('Prédiction', fontsize=12)
    ax.set_ylabel('Vérité', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Matrice de confusion sauvegardée: {save_path}")
    
    return fig


# =============================================================================
# 5. INTERPRETABILITY
# =============================================================================

def setup_interpretability(
    results_dir: Path,
    verbose: bool = True
) -> Tuple[Any, Any, Any, Any]:
    """
    Import and setup interpretability modules.
    
    Args:
        results_dir: Directory to save results
        verbose: Print setup information
    
    Returns:
        Tuple of (GradCAM, LIMEImageExplainer, SHAPExplainer, plot_multiple_explanations)
    """
    if verbose:
        print("=" * 70)
        print("IMPORT DES MODULES D'INTERPRÉTABILITÉ")
        print("=" * 70)
    
    try:
        from src.interpretability import (
            GradCAM,
            LIMEImageExplainer,
            SHAPExplainer,
            plot_multiple_explanations
        )
        
        if verbose:
            print("\n✅ Modules d'interprétabilité importés")
        
        interp_dir = results_dir / 'interpretability'
        interp_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"📂 Résultats sauvegardés dans: {interp_dir}")
        
        return GradCAM, LIMEImageExplainer, SHAPExplainer, plot_multiple_explanations
    
    except ImportError as e:
        logger.error(f"Erreur d'import: {e}")
        if verbose:
            print(f"\n⚠️ Erreur d'import: {e}")
        return None, None, None, None


def run_gradcam_analysis(
    model: keras.Model,
    X_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    categories: List[str],
    sample_indices: np.ndarray,
    save_dir: Path,
    verbose: bool = True
) -> Tuple[Any, List[np.ndarray]]:
    """
    Run Grad-CAM analysis on sample images.
    
    Args:
        model: Trained model
        X_test: Test images
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        categories: Category names
        sample_indices: Indices of images to analyze
        save_dir: Directory to save results
        verbose: Print progress
    
    Returns:
        Tuple of (gradcam_explainer, heatmaps)
    """
    from src.interpretability import GradCAM
    from src.interpretability.gradcam import visualize_gradcam_grid
    
    if verbose:
        print("=" * 70)
        print("GRAD-CAM - VISUALISATION DES ZONES D'ATTENTION")
        print("=" * 70)
    
    # Create explainer
    gradcam = GradCAM(model)
    
    if verbose:
        conv_layers = gradcam.get_available_layers()
        print(f"\n📐 Couches convolutionnelles disponibles: {len(conv_layers)}")
        print(f"   Couche utilisée: {gradcam.layer_name}")
        print("\n🔍 Génération des heatmaps Grad-CAM...")
    
    # Generate heatmaps
    heatmaps = []
    sample_images = []
    sample_class_names = []
    sample_confidences = []
    
    for idx in sample_indices:
        image = X_test[idx]
        pred_label = y_pred[idx]
        confidence = y_pred_proba[idx, pred_label]
        
        heatmap = gradcam.compute_heatmap(image, class_idx=pred_label)
        
        heatmaps.append(heatmap)
        sample_images.append(image)
        sample_class_names.append(categories[pred_label])
        sample_confidences.append(confidence)
    
    # Visualize grid
    fig = visualize_gradcam_grid(
        np.array(sample_images),
        heatmaps,
        sample_class_names,
        confidences=sample_confidences,
        n_cols=3,
        figsize=(18, 12),
        save_path=save_dir / 'gradcam_grid.png'
    )
    plt.close(fig)
    
    if verbose:
        print("✅ Visualisations Grad-CAM générées")
    
    return gradcam, heatmaps


def select_sample_images(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    categories: List[str],
    n_samples: int = 6,
    random_seed: int = 42,
    verbose: bool = True
) -> np.ndarray:
    """
    Select sample images for analysis.
    
    Args:
        X_test: Test images
        y_test: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        categories: Category names
        n_samples: Number of samples to select
        random_seed: Random seed
        verbose: Print sample information
    
    Returns:
        Array of sample indices
    """
    np.random.seed(random_seed)
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    if verbose:
        print(f"\n🎯 Analyse de {n_samples} images du test set:")
        for idx in sample_indices:
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            confidence = y_pred_proba[idx, pred_label]
            
            status = "✅" if true_label == pred_label else "❌"
            print(f"  {status} Image {idx}: Vrai={categories[true_label]:15s} | "
                  f"Prédit={categories[pred_label]:15s} ({confidence:.2%})")
    
    return sample_indices
