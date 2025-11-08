"""
Script d'entraînement InceptionV3 pour Classification COVID-19

Ce script entraîne un modèle InceptionV3 avec fine-tuning en 2 phases :
1. Phase 1: Feature extraction (base gelée)
2. Phase 2: Fine-tuning (top layers dégelées)

Inclut :
- Chargement et préparation des données
- Augmentation de données
- Entraînement avec callbacks
- Évaluation complète
- Visualisations (courbes, matrices, ROC)
- Interprétabilité (Grad-CAM, LIME)
- Sauvegarde du modèle et des résultats

Author: Data Pipeline Team
Date: November 2025
"""

import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
    precision_recall_fscore_support
)

# Supprimer les warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Imports du projet

from src.notebooks import (
    # Data
    load_dataset,
    prepare_train_val_test_split,
    compute_class_weights,
    # Model
    build_transfer_learning_model,
    unfreeze_top_layers,
    compile_model,
    create_callbacks,
    # Visualization
    plot_training_curves,
    plot_confusion_matrix,
    # Interpretability
    run_full_interpretability_analysis,
    get_preprocessing_function,
)


# =============================================================================
# CONFIGURATION
# =============================================================================
"""
╔════════════════════════════════════════════════════════════════════════════╗
║  🎯 CELLULE DE CONFIGURATION STANDALONE - COPIER-COLLER DANS VOS NOTEBOOKS ║
╚════════════════════════════════════════════════════════════════════════════╝

INSTRUCTIONS:
-------------
1. Copiez TOUT le contenu de cette cellule
2. Collez-le comme PREMIÈRE CELLULE de votre notebook
3. Exécutez la cellule
4. La configuration est prête à l'emploi !

Cette cellule est 100% autonome et fonctionne partout :
✅ Google Colab (clone + installe automatiquement)
✅ WSL / Linux Local
✅ Tout environnement Jupyter

APRÈS EXÉCUTION, UTILISEZ L'OBJET 'config':
--------------------------------------------
▶ config.data_dir              # Chemin du dataset
▶ config.models_dir            # Répertoire des modèles
▶ config.results_dir           # Répertoire des résultats
▶ config.classes               # Liste des classes
▶ config.img_size              # Tuple (width, height)
▶ config.img_channels          # Nombre de canaux (1=grayscale, 3=RGB)
▶ config.batch_size            # Taille des batchs
▶ config.epochs                # Nombre d'époques
▶ config.learning_rate         # Learning rate
▶ config.validation_split      # Proportion pour validation
▶ config.gradcam_alpha         # Alpha pour Grad-CAM
▶ config.shap_max_evals        # Evaluations SHAP
▶ config.confidence_high_threshold  # Seuil confiance haute
... et bien plus !

VARIABLES GLOBALES:
-------------------
• config: Objet Config complet (tous les paramètres du projet)
• ENV: Environnement détecté ('colab', 'wsl', 'local')
• Tous les transformers importés et prêts à l'emploi

"""

# =============================================================================
# IMPORTS STANDARDS
# =============================================================================

import os
import sys
import subprocess
from pathlib import Path


# =============================================================================
# DÉTECTION AUTOMATIQUE DE L'ENVIRONNEMENT
# =============================================================================

def detect_environment():
    """Détecte l'environnement (colab, wsl, local)"""
    try:
        import google.colab
        return "colab"
    except ImportError:
        is_wsl = os.path.exists('/proc/version') and 'microsoft' in open('/proc/version').read().lower()
        return "wsl" if is_wsl else "local"

ENV = detect_environment()
print(f"🌍 Environnement: {ENV.upper()}")


# =============================================================================
# BOOTSTRAP COLAB (Clone + Install si nécessaire)
# =============================================================================

if ENV == "colab":
    print("\n🚀 Bootstrap Colab...")

    os.chdir('/content')
    if not os.path.exists('/content/Data_Pipeline'):
        print("📥 Clonage du repository...")
        subprocess.run(['git', 'clone', 'https://github.com/L-Poca/Data_Pipeline.git'], check=True)

    os.chdir('/content/Data_Pipeline')

    # Checkout de la branche rafael_cleaning
    result = subprocess.run(
        ['git', 'checkout', '-b', 'rafael_cleaning', 'origin/rafael_cleaning'],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        # Si la branche locale existe déjà, juste switcher
        subprocess.run(['git', 'checkout', 'rafael_cleaning'], capture_output=True)

    # Installation du package en mode éditable (sans dépendances - détection Colab dans setup.py)
    print("📦 Installation du package...")
    result = subprocess.run(['pip', 'install', '-e', '.', '--quiet'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ Erreur installation: {result.stderr}")
    else:
        print("✅ Package installé")

    print("💾 Montage Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive')

    # Extraction dataset
    archive_data = '/content/drive/MyDrive/DS_COVID/archive_covid.zip'
    if os.path.exists(archive_data):
        print("📦 Extraction dataset...")
        os.makedirs('./data/raw/', exist_ok=True)
        subprocess.run(['unzip', '-o', '-q', archive_data, '-d', './data/raw/COVID-19_Radiography_Dataset/'])

    # Extraction models
    archive_models = '/content/drive/MyDrive/DS_COVID/inceptionv3_best.zip'
    if os.path.exists(archive_models):
        print("📦 Extraction models...")
        os.makedirs('./models/', exist_ok=True)
        subprocess.run(['unzip', '-o', '-q', archive_models, '-d', './models/'])

    print("✅ Bootstrap terminé")


# =============================================================================
# CONFIGURATION DES CHEMINS
# =============================================================================

# Déterminer project_root selon l'environnement
if ENV == "colab":
    project_root = Path('/content/Data_Pipeline')
elif ENV == "wsl":
    project_root = Path('/home/cepa/DST/projet_DS/Data_Pipeline/Data_Pipeline')
    #project_root = Path.cwd().parent.parent
else:  # local
    # Depuis un notebook dans src/notebooks/
    project_root = Path.cwd().parent.parent

# Vérification du modèle en local (WSL ou autre)
if ENV != "colab":
    models_dir = project_root / 'models'
    model_path = models_dir / 'inceptionv3_best.keras'

    if model_path.exists():
        print(f"✅ Modèle InceptionV3 trouvé: {model_path}")
    else:
        print(f"⚠️ Modèle InceptionV3 non trouvé: {model_path}")
        print(f"   Veuillez placer inceptionv3_best.keras dans {models_dir}/")

# Ajouter src/ au sys.path pour les imports
# src_path = str(project_root / 'src')
# if src_path not in sys.path:
#     sys.path.insert(0, src_path)
#     print(f"✅ Chemin src/ ajouté: {src_path}")

# Charger la configuration depuis JSON
from src.utils.config import build_config

config = build_config(project_root, ENV)

print(f"\n🎯 Configuration chargée depuis config/{ENV}_config.json")


# =============================================================================
# IMPORTS DES TRANSFORMERS
# =============================================================================

try:
    from src.features.Pipelines.Transformateurs.image_loaders import ImageLoader
    from src.features.Pipelines.Transformateurs.image_preprocessing import (
        ImageResizer, ImageNormalizer, ImageFlattener, ImageMasker
    )
    from src.features.Pipelines.Transformateurs.image_augmentation import (
        ImageAugmenter, ImageRandomCropper
    )
    from src.features.Pipelines.Transformateurs.image_features import (
        ImageHistogram, ImagePCA, ImageStandardScaler
    )
    print("✅ Transformers importés")
except ImportError as e:
    print(f"⚠️ Erreur import transformers: {e}")


# =============================================================================
# IMPORTS ML/DL
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# =============================================================================
# CONFIGURATION MATPLOTLIB (utilise config pour les paramètres)
# =============================================================================

plt.rcParams['figure.figsize'] = config.figure_size
plt.rcParams['figure.dpi'] = config.dpi
plt.style.use(config.plot_style)
sns.set_palette(config.color_palette)

# =============================================================================
# AFFICHAGE DU RÉSUMÉ
# =============================================================================

print("\n" + "=" * 80)
print("✅ CONFIGURATION PRÊTE - Data Pipeline")
print("=" * 80)
print(f"📂 Projet:       {config.project_root}")
print(f"📊 Dataset:      {config.data_dir}")
print(f"💾 Modèles:      {config.models_dir}")
print(f"📈 Résultats:    {config.results_dir}")
print(f"📐 Dataset:      {'✅ Accessible' if config.data_dir.exists() else '❌ Introuvable'}")
print()
print(f"🏷️  Classes:     {', '.join(config.classes)} ({config.num_classes} classes)")
print(f"🎛️  Images:      {config.img_size} | {config.img_channels} canaux")
print(f"🔧 Training:     Batch={config.batch_size} | Epochs={config.epochs} | LR={config.learning_rate}")
print(f"� Splits:       Train/Val={1-config.validation_split:.0%} | Val={config.validation_split:.0%} | Test={config.test_split:.0%}")
print()
print(f"🎨 Viz:          Style={config.plot_style} | Palette={config.color_palette}")
print(f"📏 Figures:      {config.figure_size} @ {config.dpi} DPI")
print()
print(f"🔍 Interprét.:   GradCAM α={config.gradcam_alpha} | SHAP evals={config.shap_max_evals}")
print(f"📉 Seuils conf.: High={config.confidence_high_threshold} | Medium={config.confidence_medium_threshold}")
print("=" * 80)
print("\n💡 Variable principale:")
print("   • config: Objet Config complet (accès à TOUS les paramètres)")
print("   • ENV: Environnement actuel")
print()
print("📚 Exemples d'utilisation:")
print("   config.data_dir          # Chemin du dataset")
print("   config.classes           # Liste des classes")
print("   config.img_size          # Tuple (width, height)")
print("   config.batch_size        # Taille des batchs")
print("   config.models_dir        # Répertoire des modèles")
print("   config.gradcam_alpha     # Paramètres d'interprétabilité")
print()
print("🎯 Transformers disponibles:")
print("   • ImageLoader, ImageResizer, ImageNormalizer, ImageFlattener, ImageMasker")
print("   • ImageAugmenter, ImageRandomCropper")
print("   • ImageHistogram, ImagePCA, ImageStandardScaler")
print("=" * 80)



# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

# =============================================================================
# CONFIGURATION DU DATASET
# =============================================================================

print("=" * 70)
print("CONFIGURATION DU DATASET")
print("=" * 70)

# ⚠️ PARAMÈTRE IMPORTANT : Nombre d'images par classe
# None = toutes les images (~21K total)
# 100/500/1000 = tests rapides
N_IMAGES_PER_CLASS = 100 #None # Modifier pour tests rapides

print(f"\n📊 Configuration:")
print(f"   Images par classe: {N_IMAGES_PER_CLASS if N_IMAGES_PER_CLASS else 'TOUTES'}")

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

print("\n" + "=" * 70)
print("CHARGEMENT DES DONNÉES")
print("=" * 70)

from src.notebooks import load_dataset, create_preprocessing_pipeline

# Charger les chemins des images
image_paths, mask_paths, labels, labels_int = load_dataset(
    data_dir=config.data_dir,
    categories=config.classes,
    n_images_per_class=N_IMAGES_PER_CLASS,
    load_masks=False,  # Pas besoin des masques pour classification
    verbose=True
)

print(f"\n✅ Dataset chargé:")
print(f"   Total images: {len(image_paths)}")
print(f"   Classes: {config.classes}")
print(f"   Distribution: {np.bincount(labels_int)}")

# Créer pipeline de preprocessing
pipeline_img = create_preprocessing_pipeline(
    img_size=config.img_size,
    color_mode='L',  # Grayscale
    mask_paths=None,
    verbose=True
)

# Charger et preprocesser
print("\n📊 Preprocessing des images...")
images = pipeline_img.fit_transform(image_paths)
images = images.astype('float32') / 255.0  # Normaliser [0, 1]

print(f"\n✅ Images préparées:")
print(f"   Shape: {images.shape}")
print(f"   Range: [{images.min():.3f}, {images.max():.3f}]")

# Visualisation échantillons
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i in range(4):
    for j in range(8):
        idx = i * (len(images) // 4) + j
        if idx < len(images):
            axes[i, j].imshow(images[idx], cmap='gray')
            if j == 0:
                axes[i, j].set_ylabel(config.classes[i], rotation=0, ha='right', va='center')
            axes[i, j].axis('off')
plt.suptitle('Échantillons du Dataset', size=14, weight='bold')
plt.tight_layout()
plt.show()

# =============================================================================
# ANALYSE DU CLASS IMBALANCE
# =============================================================================

print("=" * 70)
print("ANALYSE DU CLASS IMBALANCE")
print("=" * 70)

# Distribution
unique, counts = np.unique(labels_int, return_counts=True)
total = len(labels_int)

print("\n📊 Distribution actuelle:")
for cls_idx, count in zip(unique, counts):
    percentage = (count / total) * 100
    print(f"   {config.classes[cls_idx]:20s}: {count:6d} images ({percentage:5.2f}%)")

# Ratio de déséquilibre
max_count, min_count = counts.max(), counts.min()
imbalance_ratio = max_count / min_count
print(f"\n⚠️ Ratio de déséquilibre: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 2:
    print("   → Class imbalance significatif détecté!")
    print("   → Stratégies de rééquilibrage nécessaires")

# Visualisation
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette('husl', len(config.classes))
bars = ax.bar(config.classes, counts, color=colors)
ax.set_ylabel('Nombre d\'images', fontsize=11)
ax.set_title('Distribution des Classes (Déséquilibrée)', fontsize=13, weight='bold')
ax.grid(axis='y', alpha=0.3)

# Ajouter pourcentages sur les barres
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({count/total*100:.1f}%)',
            ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

# =============================================================================
# STRATÉGIES DE RÉÉQUILIBRAGE
# =============================================================================

print("\n" + "=" * 70)
print("STRATÉGIES DE RÉÉQUILIBRAGE")
print("=" * 70)

imbalance_strategies = {}

# 1. CLASS WEIGHTS (sklearn)
print("\n1️⃣ Class Weights (sklearn)")
from sklearn.utils.class_weight import compute_class_weight

class_weights_array = compute_class_weight(
    'balanced',
    classes=np.unique(labels_int),
    y=labels_int
)
class_weights_dict = dict(enumerate(class_weights_array))

print("   Poids calculés:")
for cls_idx, weight in class_weights_dict.items():
    print(f"      {config.classes[cls_idx]:20s}: {weight:.3f}")

imbalance_strategies['class_weights'] = class_weights_dict

# 2. SMOTE (si disponible)
if IMBLEARN_AVAILABLE:
    print("\n2️⃣ SMOTE (Synthetic Minority Over-sampling)")
    print("   ✅ Disponible (sera appliqué lors du training ML)")
    imbalance_strategies['smote_available'] = True
else:
    print("\n2️⃣ SMOTE non disponible")
    imbalance_strategies['smote_available'] = False

# 3. RANDOM OVERSAMPLING
print("\n3️⃣ Random Oversampling")
print("   ✅ Disponible (duplication d'images de la classe minoritaire)")
imbalance_strategies['oversampling'] = True

# 4. RANDOM UNDERSAMPLING
print("\n4️⃣ Random Undersampling")
print("   ✅ Disponible (réduction de la classe majoritaire)")
imbalance_strategies['undersampling'] = True

print("\n✅ Stratégies identifiées et prêtes")

# =============================================================================
# SPLIT TRAIN/VAL/TEST STRATIFIÉ
# =============================================================================

print("=" * 70)
print("SPLIT TRAIN/VAL/TEST STRATIFIÉ")
print("=" * 70)

# Étape 1: Split 70/30 (train / temp)
X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels_int,
    test_size=0.30,
    random_state=config.random_seed,
    stratify=labels_int
)

# Étape 2: Split 30 → 15/15 (validation / test)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=config.random_seed,
    stratify=y_temp
)

print(f"\n📊 Splits créés:")
print(f"   Train: {X_train.shape[0]:5d} images ({X_train.shape[0]/len(images)*100:.1f}%)")
print(f"      Distribution: {np.bincount(y_train)}")
print(f"   Val:   {X_val.shape[0]:5d} images ({X_val.shape[0]/len(images)*100:.1f}%)")
print(f"      Distribution: {np.bincount(y_val)}")
print(f"   Test:  {X_test.shape[0]:5d} images ({X_test.shape[0]/len(images)*100:.1f}%)")
print(f"      Distribution: {np.bincount(y_test)}")

# Vérifier stratification
print("\n✅ Vérification de la stratification:")
print(f"   {'Classe':<20s} {'Train %':>10s} {'Val %':>10s} {'Test %':>10s}")
print("   " + "-" * 50)
for i, cls_name in enumerate(config.classes):
    train_pct = (y_train == i).sum() / len(y_train) * 100
    val_pct = (y_val == i).sum() / len(y_val) * 100
    test_pct = (y_test == i).sum() / len(y_test) * 100
    print(f"   {cls_name:<20s} {train_pct:>9.2f}% {val_pct:>9.2f}% {test_pct:>9.2f}%")

# =============================================================================
# DATA AUGMENTATION
# =============================================================================

print("=" * 70)
print("DATA AUGMENTATION")
print("=" * 70)

# Configuration
augmentation_config = {
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip': True,
    'zoom_range': 0.15,
    'shear_range': 0.1,
    'fill_mode': 'nearest'
}

print("\n🔧 Configuration:")
for key, value in augmentation_config.items():
    print(f"   {key:25s}: {value}")

# Créer générateurs
train_datagen = ImageDataGenerator(**augmentation_config)
val_datagen = ImageDataGenerator()  # Pas d'augmentation pour validation

print("\n✅ Générateurs créés")

# Visualisation de l'effet
print("\n📸 Visualisation de l'augmentation...")
sample_img = X_train[0:1]
if sample_img.ndim == 3:
    sample_img = sample_img[..., np.newaxis]

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.ravel()

axes[0].imshow(sample_img[0, :, :, 0], cmap='gray')
axes[0].set_title('Original', fontsize=10)
axes[0].axis('off')

aug_iter = train_datagen.flow(sample_img, batch_size=1)
for i in range(1, 9):
    aug_img = next(aug_iter)[0]
    axes[i].imshow(aug_img[:, :, 0], cmap='gray')
    axes[i].set_title(f'Augmentée {i}', fontsize=10)
    axes[i].axis('off')

plt.suptitle('Effet de l\'Augmentation de Données', size=14, weight='bold')
plt.tight_layout()
plt.show()

# =============================================================================
# ENTRAÎNEMENT PHASE 1 : FEATURE EXTRACTION
# =============================================================================

def train_phase1(config, X_train, y_train, X_val, y_val, class_weights_dict):
    """
    Phase 1: Feature extraction avec base gelée.
    
    Returns:
        Tuple de (model, base_model, history)
    """
    print("\n" + "=" * 70)
    print("PHASE 1: FEATURE EXTRACTION (Base gelée)")
    print("=" * 70)
    
    # Construire le modèle
    print("\n🔨 Construction du modèle InceptionV3...")
    model, base_model = build_transfer_learning_model(
        base_model_name='InceptionV3',
        input_shape=(config.img_size[0], config.img_size[1], 3),
        num_classes=config.num_classes,
        freeze_base=True,
        verbose=True
    )
    
    # Compiler
    print("\n⚙️  Compilation du modèle...")
    model = compile_model(model, learning_rate=config.learning_rate, verbose=True)
    
    # Callbacks
    model_save_dir_p1 = config.models_dir / "InceptionV3_phase1"
    callbacks_phase1 = create_callbacks(
        models_dir=model_save_dir_p1,
        patience_early_stop=10,
        patience_reduce_lr=5,
        monitor='val_accuracy',
        verbose=True
    )
    
    # Entraînement
    print("\n🚀 Entraînement Phase 1...")
    print(f"   • Epochs: 20")
    print(f"   • Batch size: {config.batch_size}")
    
    start_time = time.time()
    
    history_phase1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=config.batch_size,
        callbacks=callbacks_phase1,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    train_time_p1 = time.time() - start_time
    
    print(f"\n✅ Phase 1 terminée en {train_time_p1:.2f} secondes")
    print(f"   • Best val_accuracy: {max(history_phase1.history['val_accuracy']):.4f}")
    
    return model, base_model, history_phase1, train_time_p1


# =============================================================================
# ENTRAÎNEMENT PHASE 2 : FINE-TUNING
# =============================================================================

def train_phase2(config, model, base_model, X_train, y_train, X_val, y_val, class_weights_dict):
    """
    Phase 2: Fine-tuning avec top layers dégelées.
    
    Returns:
        Tuple de (model, history)
    """
    print("\n" + "=" * 70)
    print("PHASE 2: FINE-TUNING (Top layers dégelées)")
    print("=" * 70)
    
    # Dégeler les top layers
    print("\n🔓 Dégel des top layers...")
    model = unfreeze_top_layers(
        base_model=base_model,
        model=model,
        n_layers=30,
        learning_rate=config.learning_rate / 10,
        verbose=True
    )
    
    # Callbacks
    model_save_dir_p2 = config.models_dir / "InceptionV3_phase2"
    callbacks_phase2 = create_callbacks(
        models_dir=model_save_dir_p2,
        patience_early_stop=15,
        patience_reduce_lr=7,
        monitor='val_accuracy',
        verbose=True
    )
    
    # Entraînement
    print("\n🚀 Entraînement Phase 2...")
    print(f"   • Epochs: {config.epochs}")
    print(f"   • Batch size: {config.batch_size}")
    print(f"   • Learning rate: {config.learning_rate / 10}")
    
    start_time = time.time()
    
    history_phase2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks_phase2,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    train_time_p2 = time.time() - start_time
    
    print(f"\n✅ Phase 2 terminée en {train_time_p2:.2f} secondes")
    print(f"   • Best val_accuracy: {max(history_phase2.history['val_accuracy']):.4f}")
    
    return model, history_phase2, train_time_p2


# =============================================================================
# ÉVALUATION
# =============================================================================

def evaluate_model(config, model, X_test, y_test):
    """
    Évalue le modèle sur le test set.
    
    Returns:
        Dict avec toutes les métriques
    """
    print("\n" + "=" * 70)
    print("ÉVALUATION SUR LE TEST SET")
    print("=" * 70)
    
    # Prédictions
    print("\n🔮 Prédictions...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Métriques de base
    eval_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss = eval_results[0]
    test_acc = eval_results[1]
    
    # F1 Score
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    # Precision, Recall
    precision, recall, _, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    print(f"\n📊 Résultats:")
    print(f"   • Test Loss: {test_loss:.4f}")
    print(f"   • Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   • F1 Score (weighted): {f1_weighted:.4f}")
    print(f"   • Precision (weighted): {precision:.4f}")
    print(f"   • Recall (weighted): {recall:.4f}")
    
    print(f"\n📋 F1 Score par classe:")
    for i, class_name in enumerate(config.classes):
        print(f"   • {class_name}: {f1_per_class[i]:.4f}")
    
    # Rapport de classification
    print(f"\n📄 Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=config.classes,
        digits=4
    ))
    
    results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'precision': precision,
        'recall': recall,
        'y_pred': y_pred,
        'y_pred_probs': y_pred_probs
    }
    
    return results


# =============================================================================
# VISUALISATIONS
# =============================================================================

def create_visualizations(config, history_phase1, history_phase2, y_test, y_pred, y_pred_probs):
    """Crée toutes les visualisations."""
    print("\n" + "=" * 70)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("=" * 70)
    
    results_dir = config.results_dir / 'inceptionv3'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Courbes d'entraînement
    print("\n📈 Courbes d'entraînement...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Phase 1
    axes[0, 0].plot(history_phase1.history['loss'], label='Train Loss')
    axes[0, 0].plot(history_phase1.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Phase 1: Loss', fontsize=12, weight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(history_phase1.history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history_phase1.history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Phase 1: Accuracy', fontsize=12, weight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Phase 2
    axes[1, 0].plot(history_phase2.history['loss'], label='Train Loss')
    axes[1, 0].plot(history_phase2.history['val_loss'], label='Val Loss')
    axes[1, 0].set_title('Phase 2: Loss', fontsize=12, weight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(history_phase2.history['accuracy'], label='Train Accuracy')
    axes[1, 1].plot(history_phase2.history['val_accuracy'], label='Val Accuracy')
    axes[1, 1].set_title('Phase 2: Accuracy', fontsize=12, weight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('InceptionV3 - Training Curves', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Sauvegardé: training_curves.png")
    
    # 2. Matrice de confusion
    print("\n📊 Matrice de confusion...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=config.classes,
        yticklabels=config.classes
    )
    plt.title('InceptionV3 - Confusion Matrix', fontsize=14, weight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Sauvegardé: confusion_matrix.png")
    
    # 3. Courbes ROC
    print("\n📉 Courbes ROC...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, class_name in enumerate(config.classes):
        # Binariser les labels
        y_test_binary = (y_test == i).astype(int)
        y_pred_binary = y_pred_probs[:, i]
        
        # Calculer ROC
        fpr, tpr, _ = roc_curve(y_test_binary, y_pred_binary)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        axes[i].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', linewidth=2)
        axes[i].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[i].set_title(f'{class_name}', fontsize=12, weight='bold')
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    
    plt.suptitle('InceptionV3 - ROC Curves', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Sauvegardé: roc_curves.png")
    
    print(f"\n✅ Visualisations sauvegardées dans: {results_dir}")


# =============================================================================
# INTERPRÉTABILITÉ
# =============================================================================

def run_interpretability(config, model, X_test, y_test, y_pred):
    """Exécute l'analyse d'interprétabilité."""
    print("\n" + "=" * 70)
    print("ANALYSE D'INTERPRÉTABILITÉ")
    print("=" * 70)
    
    interpretability_dir = config.results_dir / 'inceptionv3' / 'interpretability'
    interpretability_dir.mkdir(parents=True, exist_ok=True)
    
    # Preprocessing function pour InceptionV3
    try:
        preprocess_fn = get_preprocessing_function('InceptionV3')
        print("   ✅ Preprocessing function: InceptionV3")
    except Exception as e:
        print(f"   ⚠️  Preprocessing function non trouvée: {e}")
        preprocess_fn = None
    
    # Analyse complète
    try:
        print("\n🔍 Lancement de l'analyse (Grad-CAM + LIME)...")
        
        run_full_interpretability_analysis(
            model=model,
            x_data=X_test,
            y_true=y_test,
            y_pred=y_pred,
            class_names=config.classes,
            background_data=X_test[:50],
            n_samples=2,
            strategy="one_per_class",
            save_dir=interpretability_dir,
            use_gradcam=True,
            use_lime=True,
            use_shap=False,
            preprocess_fn=preprocess_fn
        )
        
        print(f"\n✅ Interprétabilité sauvegardée dans: {interpretability_dir}")
        
    except Exception as e:
        print(f"⚠️  Erreur lors de l'analyse d'interprétabilité: {e}")
        print("   Continuez sans interprétabilité ou installez les dépendances:")
        print("   pip install lime shap")


# =============================================================================
# SAUVEGARDE
# =============================================================================

def save_model_and_results(config, model, results, train_time_p1, train_time_p2):
    """Sauvegarde le modèle et les résultats."""
    print("\n" + "=" * 70)
    print("SAUVEGARDE")
    print("=" * 70)
    
    # Sauvegarder le modèle
    model_path = config.models_dir / 'inceptionv3_final.keras'
    model.save(model_path)
    print(f"\n✅ Modèle sauvegardé: {model_path}")
    
    # Sauvegarder les résultats
    results_file = config.results_dir / 'inceptionv3' / 'results.txt'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("INCEPTIONV3 - RÉSULTATS FINAUX\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  • Image size: {config.img_size}\n")
        f.write(f"  • Batch size: {config.batch_size}\n")
        f.write(f"  • Learning rate: {config.learning_rate}\n")
        f.write(f"  • Epochs Phase 1: 20\n")
        f.write(f"  • Epochs Phase 2: {config.epochs}\n\n")
        
        f.write("Temps d'entraînement:\n")
        f.write(f"  • Phase 1: {train_time_p1:.2f}s\n")
        f.write(f"  • Phase 2: {train_time_p2:.2f}s\n")
        f.write(f"  • Total: {train_time_p1 + train_time_p2:.2f}s\n\n")
        
        f.write("Métriques:\n")
        f.write(f"  • Test Loss: {results['test_loss']:.4f}\n")
        f.write(f"  • Test Accuracy: {results['test_acc']:.4f} ({results['test_acc']*100:.2f}%)\n")
        f.write(f"  • F1 Score (weighted): {results['f1_weighted']:.4f}\n")
        f.write(f"  • Precision (weighted): {results['precision']:.4f}\n")
        f.write(f"  • Recall (weighted): {results['recall']:.4f}\n\n")
        
        f.write("F1 Score par classe:\n")
        for i, class_name in enumerate(config.classes):
            f.write(f"  • {class_name}: {results['f1_per_class'][i]:.4f}\n")
    
    print(f"✅ Résultats sauvegardés: {results_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Fonction principale."""
    print("\n" + "=" * 70)
    print("ENTRAÎNEMENT INCEPTIONV3 - COVID-19 CLASSIFICATION")
    print("=" * 70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Configuration
    config = setup_environment()
    
    # 2. Chargement des données
    # Mettre n_images_per_class=None pour utiliser tout le dataset
    X_train, X_val, X_test, y_train, y_val, y_test, class_weights_dict = load_and_prepare_data(
        config,
        n_images_per_class=None  # None = toutes les images
    )
    
    # Convertir les labels en categorical
    from tensorflow.keras.utils import to_categorical
    y_train_cat = to_categorical(y_train, num_classes=config.num_classes)
    y_val_cat = to_categorical(y_val, num_classes=config.num_classes)
    y_test_cat = to_categorical(y_test, num_classes=config.num_classes)
    
    # 3. Entraînement Phase 1
    model, base_model, history_phase1, train_time_p1 = train_phase1(
        config, X_train, y_train_cat, X_val, y_val_cat, class_weights_dict
    )
    
    # 4. Entraînement Phase 2
    model, history_phase2, train_time_p2 = train_phase2(
        config, model, base_model, X_train, y_train_cat, X_val, y_val_cat, class_weights_dict
    )
    
    # 5. Évaluation
    results = evaluate_model(config, model, X_test, y_test_cat)
    
    # 6. Visualisations
    create_visualizations(
        config, history_phase1, history_phase2,
        y_test, results['y_pred'], results['y_pred_probs']
    )
    
    # 7. Interprétabilité
    run_interpretability(config, model, X_test, y_test, results['y_pred'])
    
    # 8. Sauvegarde
    save_model_and_results(config, model, results, train_time_p1, train_time_p2)
    
    # Résumé final
    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 70)
    print(f"\n📊 Résultats finaux:")
    print(f"   • Test Accuracy: {results['test_acc']:.4f} ({results['test_acc']*100:.2f}%)")
    print(f"   • F1 Score: {results['f1_weighted']:.4f}")
    print(f"   • Temps total: {train_time_p1 + train_time_p2:.2f}s")
    
    print(f"\n📁 Fichiers générés:")
    print(f"   • Modèle: {config.models_dir / 'inceptionv3_final.keras'}")
    print(f"   • Résultats: {config.results_dir / 'inceptionv3' / 'results.txt'}")
    print(f"   • Visualisations: {config.results_dir / 'inceptionv3'}")
    print(f"   • Interprétabilité: {config.results_dir / 'inceptionv3' / 'interpretability'}")
    
    print("\n🎉 Script terminé avec succès !")


if __name__ == "__main__":
    main()
