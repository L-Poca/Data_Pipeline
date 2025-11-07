# 🎯 Guide d'utilisation de l'objet Config

## Vue d'ensemble

L'objet `config` est le **point d'entrée unique** pour tous les paramètres du projet. Il remplace les variables dispersées et hardcodées dans le code.

## 🚀 Quick Start

### 1. Dans vos notebooks

```python
# Copier-coller CELL_CONFIG_STANDALONE.py en première cellule
# Après exécution, l'objet config est disponible globalement

# ✅ Utilisation directe
model.fit(
    X_train, y_train,
    batch_size=config.batch_size,
    epochs=config.epochs,
    validation_split=config.validation_split
)
```

### 2. Structure de config

```python
config.
├── Chemins
│   ├── project_root
│   ├── data_dir
│   ├── models_dir
│   └── results_dir
├── Images
│   ├── img_width, img_height, img_size
│   └── img_channels
├── Training
│   ├── batch_size, epochs, learning_rate
│   ├── validation_split, test_split
│   └── random_seed
├── Modèles ML
│   ├── rf_n_estimators, rf_max_depth, ...
│   ├── xgb_n_estimators, xgb_learning_rate, ...
│   └── pretrained_weights, freeze_base_layers, ...
├── Callbacks
│   ├── early_stopping_patience
│   ├── reduce_lr_patience, reduce_lr_factor
│   └── min_lr
├── Visualisation
│   ├── plot_style, color_palette
│   └── figure_size, dpi
├── Interprétabilité
│   ├── gradcam_alpha, gradcam_colormap
│   ├── shap_max_evals, shap_background_size
│   └── confidence_high_threshold, confidence_medium_threshold
└── Classes & Metadata
    ├── classes
    └── num_classes (dérivé)
```

## 📚 Exemples d'utilisation

### Chemins

```python
# ✅ Direct
dataset_path = config.data_dir / "COVID"
model_path = config.models_dir / "my_model.keras"

# ❌ Éviter
data_dir = config.data_dir  # Variable intermédiaire inutile
dataset_path = data_dir / "COVID"
```

### Training

```python
# ✅ Paramètres depuis config
history = model.fit(
    X_train, y_train,
    batch_size=config.batch_size,
    epochs=config.epochs,
    validation_split=config.validation_split
)

# ❌ Éviter les hardcoded values
history = model.fit(X_train, y_train, batch_size=32, epochs=50)
```

### Callbacks

```python
# ✅ Configuration centralisée
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(patience=config.early_stopping_patience),
    ReduceLROnPlateau(
        patience=config.reduce_lr_patience,
        factor=config.reduce_lr_factor,
        min_lr=config.min_lr
    )
]
```

### Visualisation

```python
# ✅ Matplotlib configuré automatiquement
# Les paramètres sont appliqués dans CELL_CONFIG_STANDALONE

# Mais disponibles pour usage explicite
fig, ax = plt.subplots(figsize=config.figure_size, dpi=config.dpi)
```

### Interprétabilité

```python
# ✅ Grad-CAM avec config
gradcam = GradCAM(
    model=model,
    alpha=config.gradcam_alpha,
    colormap=config.gradcam_colormap
)

# Classification de confiance
max_prob = prediction.max()
if max_prob >= config.confidence_high_threshold:
    confidence = "HIGH"
elif max_prob >= config.confidence_medium_threshold:
    confidence = "MEDIUM"
else:
    confidence = "LOW"
```

## 🔧 Configuration

### Fichiers de configuration

1. **`config/default_config.json`**: Configuration par défaut (WSL/Local)
2. **`config/colab_config.json`**: Surcharges pour Google Colab

### Modification des paramètres

```json
// config/default_config.json
{
  "training": {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001
  }
}
```

```json
// config/colab_config.json (surcharge)
{
  "training": {
    "batch_size": 64,  // Override pour Colab
    "epochs": 100
  }
}
```

### Merge automatique

- Sur Colab: `default_config.json` + `colab_config.json`
- Sur WSL/Local: `default_config.json` uniquement

## 💡 Bonnes pratiques

### ✅ DO

1. **Toujours utiliser config directement**
   ```python
   model.fit(batch_size=config.batch_size)
   ```

2. **Modifier la config dans les JSON**
   ```json
   {"training": {"batch_size": 64}}
   ```

3. **Documenter les changements de config**
   ```python
   # Modification: epochs=100 pour convergence complète
   ```

4. **Utiliser config pour les chemins**
   ```python
   results_path = config.results_dir / "experiment_1"
   ```

### ❌ DON'T

1. **Ne pas créer de variables intermédiaires**
   ```python
   batch_size = config.batch_size  # ❌ Inutile
   model.fit(batch_size=batch_size)
   ```

2. **Ne pas hardcoder les valeurs**
   ```python
   model.fit(batch_size=32, epochs=50)  # ❌
   ```

3. **Ne pas modifier config.py directement**
   - Utiliser les fichiers JSON

4. **Ne pas dupliquer les paramètres**
   ```python
   BATCH_SIZE = 32  # ❌ Existe déjà dans config
   ```

## 🎓 Exemples complets

Voir `examples/config_usage_examples.py` pour des exemples détaillés :
- Pipelines scikit-learn
- Training TensorFlow/Keras
- Interprétabilité (Grad-CAM, SHAP)
- Sauvegarde et export

## 🔍 Inspection de config

```python
# Voir tous les paramètres
print(config.to_dict())

# Autocomplétion dans Jupyter
config.<TAB>

# Afficher la configuration actuelle
from pprint import pprint
pprint(config.to_dict())
```

## 📊 Avantages de cette approche

1. **Centralisation**: Un seul point de vérité pour tous les paramètres
2. **Flexibilité**: Adaptation automatique Colab/WSL via JSON
3. **Reproductibilité**: Config partageable et versionnable
4. **Maintenabilité**: Changements faciles sans modifier le code
5. **Documentation**: Tous les paramètres visibles et commentés

## 🚦 Migration d'ancien code

```python
# ❌ AVANT
data_dir = Path("data/raw")
batch_size = 32
epochs = 50
img_width, img_height = 256, 256

# ✅ APRÈS
# Plus besoin ! Tout est dans config
config.data_dir
config.batch_size
config.epochs
config.img_size  # (256, 256)
```

## 📞 Support

- Documentation complète: `CELL_CONFIG_STANDALONE.py` (docstring)
- Exemples: `examples/config_usage_examples.py`
- Config files: `config/default_config.json`, `config/colab_config.json`
- Source: `src/utils/config.py`
