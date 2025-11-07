"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  📚 EXEMPLES D'UTILISATION DE L'OBJET CONFIG                              ║
╚═══════════════════════════════════════════════════════════════════════════╝

Ce fichier montre comment tirer pleinement parti de l'objet `config` 
après avoir exécuté CELL_CONFIG_STANDALONE.py dans votre notebook.
"""

# =============================================================================
# 1️⃣ ACCÈS AUX CHEMINS (plus besoin de variables séparées!)
# =============================================================================

# ❌ AVANT (anciennes pratiques)
# data_dir = config.data_dir
# models_dir = config.models_dir
# 
# dataset_path = data_dir / "COVID"
# model_path = models_dir / "inceptionv3_best.keras"

# ✅ MAINTENANT (direct depuis config)
dataset_path = config.data_dir / "COVID"
model_path = config.models_dir / "inceptionv3_best.keras"
results_path = config.results_dir / "my_experiment"


# =============================================================================
# 2️⃣ PARAMÈTRES D'ENTRAÎNEMENT
# =============================================================================

# ✅ Utiliser directement dans le code
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(
        patience=config.early_stopping_patience,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        patience=config.reduce_lr_patience,
        factor=config.reduce_lr_factor,
        min_lr=config.min_lr
    )
]

# Compiler le modèle
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entraîner
history = model.fit(
    X_train, y_train,
    batch_size=config.batch_size,
    epochs=config.epochs,
    validation_split=config.validation_split,
    callbacks=callbacks,
    verbose=config.verbose
)


# =============================================================================
# 3️⃣ CONSTRUCTION DE PIPELINES
# =============================================================================

from sklearn.pipeline import Pipeline

# ✅ Utiliser config pour les paramètres des transformers
pipeline = Pipeline([
    ('loader', ImageLoader(data_dir=config.data_dir)),
    ('resizer', ImageResizer(target_size=config.img_size)),
    ('normalizer', ImageNormalizer(method=config.normalize_method)),
    ('pca', ImagePCA(n_components=config.pca_n_components))
])


# =============================================================================
# 4️⃣ AUGMENTATION DE DONNÉES
# =============================================================================

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Paramètres d'augmentation depuis config
datagen = ImageDataGenerator(
    rotation_range=config.augmentation_rotation_range,
    brightness_range=config.augmentation_brightness_range,
    zoom_range=config.augmentation_zoom_range,
    horizontal_flip=config.augmentation_flip_horizontal,
    vertical_flip=config.augmentation_flip_vertical
)


# =============================================================================
# 5️⃣ CONFIGURATION DE MATPLOTLIB
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Les paramètres sont déjà appliqués automatiquement dans CELL_CONFIG_STANDALONE
# Mais vous pouvez les utiliser explicitement si besoin:

fig, axes = plt.subplots(2, 2, figsize=config.figure_size, dpi=config.dpi)
# Le reste de votre code de visualisation...


# =============================================================================
# 6️⃣ INTERPRÉTABILITÉ (Grad-CAM, SHAP)
# =============================================================================

from src.interpretability.gradcam import GradCAM

# ✅ Utiliser les paramètres d'interprétabilité
gradcam = GradCAM(
    model=model,
    layer_name='conv5_block3_out',  # ou config.gradcam_layer_auto pour auto-détection
    alpha=config.gradcam_alpha,
    colormap=config.gradcam_colormap
)

# Classifier par niveau de confiance
def classify_confidence(prediction):
    max_prob = prediction.max()
    if max_prob >= config.confidence_high_threshold:
        return "HIGH"
    elif max_prob >= config.confidence_medium_threshold:
        return "MEDIUM"
    else:
        return "LOW"


# =============================================================================
# 7️⃣ SAUVEGARDE DE MODÈLES
# =============================================================================

# ✅ Format de sauvegarde configuré
model_save_path = config.models_dir / f"my_model.{config.model_save_format}"
model.save(model_save_path)


# =============================================================================
# 8️⃣ EXPORT DES RÉSULTATS
# =============================================================================

import pandas as pd

# ✅ Format d'export configuré
if config.export_predictions:
    predictions_df = pd.DataFrame({
        'image': image_paths,
        'prediction': predictions,
        'confidence': confidences
    })
    
    results_file = config.results_dir / f"predictions.{config.results_format}"
    predictions_df.to_csv(results_file, index=False)


# =============================================================================
# 9️⃣ SPLIT TRAIN/VAL/TEST
# =============================================================================

from sklearn.model_selection import train_test_split

# ✅ Utiliser les ratios depuis config
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=config.test_split + config.validation_split,
    random_state=config.random_seed,
    stratify=y
)

# Calculer le ratio val sur le reste
val_ratio = config.validation_split / (config.test_split + config.validation_split)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=1 - val_ratio,
    random_state=config.random_seed,
    stratify=y_temp
)


# =============================================================================
# 🔟 MODÈLES DE MACHINE LEARNING (Random Forest, XGBoost)
# =============================================================================

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ✅ Paramètres depuis config
rf_model = RandomForestClassifier(
    n_estimators=config.rf_n_estimators,
    max_depth=config.rf_max_depth,
    min_samples_split=config.rf_min_samples_split,
    min_samples_leaf=config.rf_min_samples_leaf,
    random_state=config.random_seed,
    n_jobs=config.n_jobs,
    verbose=config.verbose
)

xgb_model = xgb.XGBClassifier(
    n_estimators=config.xgb_n_estimators,
    learning_rate=config.xgb_learning_rate,
    max_depth=config.xgb_max_depth,
    min_child_weight=config.xgb_min_child_weight,
    random_state=config.random_seed,
    n_jobs=config.n_jobs,
    verbosity=config.verbose
)


# =============================================================================
# 💡 BONNES PRATIQUES
# =============================================================================

"""
✅ DO:
------
1. Toujours utiliser config.xxx au lieu de variables intermédiaires
2. Modifier la config dans les fichiers JSON (default_config.json ou colab_config.json)
3. Utiliser config pour tous les hyperparamètres
4. Documenter les changements de config dans vos notebooks

❌ DON'T:
---------
1. Ne pas créer de variables locales pour dupliquer config (ex: batch_size = config.batch_size)
2. Ne pas hardcoder les valeurs dans le code
3. Ne pas modifier config.py directement (utiliser les JSON)
4. Ne pas oublier que config est un objet global après CELL_CONFIG_STANDALONE

💡 ASTUCE:
----------
Pour voir TOUS les paramètres disponibles:
>>> print(config.to_dict())

Ou inspectez l'objet directement:
>>> config.<TAB>  # Autocomplétion dans Jupyter
"""


# =============================================================================
# 🎓 EXEMPLE COMPLET: Pipeline d'entraînement
# =============================================================================

def train_complete_pipeline():
    """Exemple d'utilisation complète de config dans un pipeline"""
    
    print(f"🚀 Début entraînement sur {config.num_classes} classes")
    print(f"📊 Dataset: {config.data_dir}")
    print(f"🎛️ Images: {config.img_size}, {config.img_channels} canaux")
    print(f"🔧 Batch: {config.batch_size}, Epochs: {config.epochs}")
    
    # 1. Charger les données
    from src.features.Pipelines.Transformateurs.image_loaders import ImageLoader
    loader = ImageLoader(data_dir=config.data_dir)
    X, y = loader.fit_transform(config.classes)
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_split,
        random_state=config.random_seed,
        stratify=y
    )
    
    # 3. Construire le modèle
    from src.notebooks.model_builders import build_transfer_learning_model
    model = build_transfer_learning_model(
        input_shape=(*config.img_size, config.img_channels),
        num_classes=config.num_classes,
        base_model_name='InceptionV3',
        weights=config.pretrained_weights,
        freeze_base=config.freeze_base_layers,
        fine_tune_layers=config.fine_tune_layers
    )
    
    # 4. Compiler
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 5. Callbacks
    callbacks = [
        EarlyStopping(
            patience=config.early_stopping_patience,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            patience=config.reduce_lr_patience,
            factor=config.reduce_lr_factor,
            min_lr=config.min_lr
        )
    ]
    
    # 6. Entraîner
    history = model.fit(
        X_train, y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_split=config.validation_split,
        callbacks=callbacks,
        verbose=config.verbose
    )
    
    # 7. Sauvegarder
    model_path = config.models_dir / f"model_{config.epochs}ep.{config.model_save_format}"
    model.save(model_path)
    print(f"✅ Modèle sauvegardé: {model_path}")
    
    # 8. Évaluer
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"📊 Test Accuracy: {test_acc:.4f}")
    
    return model, history


if __name__ == "__main__":
    # Pour tester ce fichier, il faut d'abord exécuter CELL_CONFIG_STANDALONE.py
    print("⚠️ Ce fichier contient uniquement des exemples.")
    print("📌 Exécutez d'abord CELL_CONFIG_STANDALONE.py dans votre notebook.")
