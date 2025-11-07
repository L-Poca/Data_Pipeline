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
