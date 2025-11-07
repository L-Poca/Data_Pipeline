#!/usr/bin/env python3
"""
Script de test pour les utilitaires notebooks.

Ce script teste toutes les fonctionnalités principales des modules:
- model_builders: construction et compilation de modèles
- data_utils: chargement et préparation des données
- training_utils: entraînement et évaluation
- visualization_utils: visualisations
- interpretability_utils: Grad-CAM

Author: Data Pipeline Team
Date: November 2025
"""

import sys
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Mode non-interactif pour les tests

# Supprimer les warnings TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from src.notebooks import (
    model_builders,
    data_utils,
    training_utils,
    visualization_utils,
    interpretability_utils
)


def test_model_builders():
    """Test des fonctions de construction de modèles."""
    print("=" * 70)
    print("TEST MODEL BUILDERS")
    print("=" * 70)
    
    # Test 1: Custom CNN
    print("\n1. Construction du modèle Custom CNN...")
    model = model_builders.build_custom_cnn(
        input_shape=(128, 128, 3),
        num_classes=4,
        verbose=False
    )
    assert model is not None
    assert model.name == "CustomCNN_COVID19"
    print(f"   ✅ Modèle créé: {len(model.layers)} layers, {model.count_params():,} params")
    
    # Test 2: Compilation
    print("\n2. Compilation du modèle...")
    model = model_builders.compile_model(model, learning_rate=0.001, verbose=False)
    assert model.optimizer is not None
    print("   ✅ Modèle compilé")
    
    # Test 3: Transfer Learning
    print("\n3. Transfer Learning - InceptionV3...")
    model_tl, base_model = model_builders.build_transfer_learning_model(
        base_model_name='InceptionV3',
        input_shape=(224, 224, 3),
        num_classes=4,
        freeze_base=True,
        verbose=False
    )
    assert model_tl is not None
    assert base_model is not None
    print(f"   ✅ Modèle TL créé: {model_tl.count_params():,} params")
    
    # Test 4: Prédiction
    print("\n4. Test de prédiction...")
    x_test = np.random.rand(2, 128, 128, 3).astype(np.float32)
    y_pred = model.predict(x_test, verbose=0)
    assert y_pred.shape == (2, 4)
    print(f"   ✅ Prédiction réussie: shape={y_pred.shape}")
    
    print("\n✅ Tous les tests model_builders réussis!\n")


def test_data_utils():
    """Test des utilitaires de données."""
    print("=" * 70)
    print("TEST DATA UTILS")
    print("=" * 70)
    
    # Test 1: Class weights
    print("\n1. Calcul des poids de classes...")
    y_train = np.array([0, 0, 1, 1, 1, 2, 2, 3])
    categories = ['COVID', 'Normal', 'Pneumonia', 'Opacity']
    class_weights = data_utils.compute_class_weights(y_train, categories, verbose=False)
    assert len(class_weights) == 4
    print(f"   ✅ Poids calculés: {len(class_weights)} classes")
    
    # Test 2: Data generators
    print("\n2. Création de générateurs de données...")
    x_train = np.random.rand(20, 128, 128, 3).astype(np.float32)
    y_train_cat = np.eye(4)[np.random.randint(0, 4, 20)]
    x_val = np.random.rand(10, 128, 128, 3).astype(np.float32)
    y_val_cat = np.eye(4)[np.random.randint(0, 4, 10)]
    
    train_gen, val_gen, test_gen = data_utils.create_data_generators(
        x_train, y_train_cat, x_val, y_val_cat,
        batch_size=4, augment_train=True, verbose=False
    )
    assert train_gen is not None
    assert val_gen is not None
    print(f"   ✅ Générateurs créés: {len(train_gen)} train batches, {len(val_gen)} val batches")
    
    # Test 3: Train/val/test split
    print("\n3. Split train/val/test...")
    images = np.random.rand(100, 128, 128, 3).astype(np.float32)
    labels_int = np.random.randint(0, 4, 100)
    
    x_tr, x_v, x_te, y_tr, y_v, y_te = data_utils.prepare_train_val_test_split(
        images, labels_int, num_classes=4,
        test_size=0.15, val_size=0.15, verbose=False
    )
    assert x_tr.shape[0] + x_v.shape[0] + x_te.shape[0] == 100
    print(f"   ✅ Split réussi: {x_tr.shape[0]} train, {x_v.shape[0]} val, {x_te.shape[0]} test")
    
    print("\n✅ Tous les tests data_utils réussis!\n")


def test_training_utils():
    """Test des utilitaires d'entraînement."""
    print("=" * 70)
    print("TEST TRAINING UTILS")
    print("=" * 70)
    
    # Créer un petit modèle pour les tests
    print("\n1. Préparation du modèle de test...")
    model = keras.Sequential([
        keras.layers.Input(shape=(32, 32, 3)),
        keras.layers.Conv2D(16, 3, activation='relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("   ✅ Modèle de test créé")
    
    # Test 2: Evaluation
    print("\n2. Test d'évaluation...")
    x_test = np.random.rand(10, 32, 32, 3).astype(np.float32)
    y_test_cat = np.eye(4)[np.random.randint(0, 4, 10)]
    
    results = training_utils.evaluate_model(
        model, (x_test, y_test_cat),
        class_names=['COVID', 'Normal', 'Pneumonia', 'Opacity'],
        verbose=False
    )
    assert 'loss' in results
    assert 'y_true' in results
    assert 'y_pred' in results
    print(f"   ✅ Évaluation réussie: {len(results)} métriques")
    
    print("\n✅ Tous les tests training_utils réussis!\n")


def test_visualization_utils():
    """Test des utilitaires de visualisation."""
    print("=" * 70)
    print("TEST VISUALIZATION UTILS")
    print("=" * 70)
    
    # Test 1: Confusion matrix
    print("\n1. Matrice de confusion...")
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 3, 3])
    class_names = ['COVID', 'Normal', 'Pneumonia', 'Opacity']
    
    import io
    sys.stdout = io.StringIO()  # Supprime les prints
    visualization_utils.plot_confusion_matrix(
        y_true, y_pred, class_names, normalize=True
    )
    sys.stdout = sys.__stdout__
    print("   ✅ Matrice de confusion générée")
    
    # Test 2: Training curves
    print("\n2. Courbes d'entraînement...")
    history = keras.callbacks.History()
    history.history = {
        'loss': [0.5, 0.4, 0.3],
        'accuracy': [0.7, 0.8, 0.85],
        'val_loss': [0.6, 0.5, 0.4],
        'val_accuracy': [0.65, 0.75, 0.8]
    }
    
    sys.stdout = io.StringIO()
    visualization_utils.plot_training_curves(history)
    sys.stdout = sys.__stdout__
    print("   ✅ Courbes d'entraînement générées")
    
    print("\n✅ Tous les tests visualization_utils réussis!\n")


def test_interpretability_utils():
    """Test des utilitaires d'interprétabilité."""
    print("=" * 70)
    print("TEST INTERPRETABILITY UTILS")
    print("=" * 70)
    
    # Test 1: Setup Grad-CAM
    print("\n1. Setup Grad-CAM...")
    model = keras.Sequential([
        keras.layers.Input(shape=(128, 128, 3)),
        keras.layers.Conv2D(32, 3, activation='relu', name='conv1'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation='relu', name='conv2'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(4, activation='softmax')
    ])
    
    gradcam = interpretability_utils.setup_interpretability(model, verbose=False)
    assert gradcam is not None
    print("   ✅ Grad-CAM initialisé")
    
    # Test 2: Compute heatmap
    print("\n2. Calcul de heatmap...")
    img = np.random.rand(128, 128, 3).astype(np.float32)
    heatmap = gradcam.compute_heatmap(img, class_idx=0)
    assert heatmap is not None
    assert len(heatmap.shape) == 2
    print(f"   ✅ Heatmap générée: shape={heatmap.shape}")
    
    # Test 3: Sample selection
    print("\n3. Sélection d'échantillons...")
    x_data = np.random.rand(50, 128, 128, 3).astype(np.float32)
    y_true = np.random.randint(0, 4, 50)
    y_pred = np.random.randint(0, 4, 50)
    class_names = ['COVID', 'Normal', 'Pneumonia', 'Opacity']
    
    sys.stdout = io.StringIO()
    indices, descriptions = interpretability_utils.select_sample_images(
        x_data, y_true, y_pred, class_names,
        n_samples=2, strategy='correct'
    )
    sys.stdout = sys.__stdout__
    assert len(indices) > 0
    print(f"   ✅ {len(indices)} échantillons sélectionnés")
    
    print("\n✅ Tous les tests interpretability_utils réussis!\n")


def main():
    """Fonction principale de test."""
    print("\n" + "=" * 70)
    print("TESTS DES UTILITAIRES NOTEBOOKS")
    print("=" * 70)
    print(f"\nKeras version: {keras.__version__}")
    print(f"NumPy version: {np.__version__}\n")
    
    try:
        test_model_builders()
        test_data_utils()
        test_training_utils()
        test_visualization_utils()
        test_interpretability_utils()
        
        print("=" * 70)
        print("✅ TOUS LES TESTS RÉUSSIS!")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERREUR LORS DES TESTS")
        print("=" * 70)
        print(f"\n{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
