# 📚 Notebook Utilities - Guide d'Utilisation

Ce module fournit des fonctions réutilisables pour simplifier les notebooks Jupyter.

## 📁 Structure

```
src/notebooks/
├── __init__.py           # Exports des fonctions
└── notebook_utils.py     # Implémentation des fonctions
```

---

## 🚀 Installation

Les fonctions sont automatiquement disponibles après installation du package :

```bash
pip install -e .
```

---

## 📖 Utilisation

### **Import simple**

```python
from src.notebooks import (
    load_dataset,
    create_preprocessing_pipeline,
    build_custom_cnn,
    train_model,
    evaluate_model,
    plot_training_curves,
    run_gradcam_analysis
)
```

---

## 🔧 Fonctions Disponibles

### **1. Data Loading & Preprocessing**

#### `load_dataset()`
Charge les chemins des images et labels depuis le dataset.

```python
image_paths, mask_paths, labels, labels_int = load_dataset(
    data_dir=Path('/path/to/data'),
    categories=['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'],
    n_images_per_class=None,  # None = toutes les images
    load_masks=False,           # True pour charger les masques
    verbose=True
)
```

**Returns:**
- `image_paths`: Liste des chemins d'images
- `mask_paths`: Liste des chemins de masques (vide si `load_masks=False`)
- `labels`: Liste des labels (strings)
- `labels_int`: Array numpy des labels (integers)

---

#### `create_preprocessing_pipeline()`
Crée une pipeline sklearn pour le preprocessing.

```python
pipeline = create_preprocessing_pipeline(
    img_size=(128, 128),
    color_mode='RGB',        # 'RGB' ou 'L' (grayscale)
    mask_paths=None,         # Optionnel: chemins des masques
    verbose=True
)

# Utiliser la pipeline
images = pipeline.fit_transform(image_paths)
images = images.astype('float32') / 255.0
```

**Étapes de la pipeline:**
1. `ImageLoader` - Charge les images
2. `ImageResizer` - Redimensionne à `img_size`
3. `ImageMasker` - Applique les masques (si `mask_paths` fourni)

---

#### `prepare_train_val_test_split()`
Split les données en train/val/test avec one-hot encoding.

```python
X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat = prepare_train_val_test_split(
    images=images,
    labels_int=labels_int,
    num_classes=4,
    test_size=0.15,
    val_size=0.15,
    random_seed=42,
    verbose=True
)
```

**Returns:**
- `X_train`, `X_val`, `X_test`: Images splitées
- `y_train_cat`, `y_val_cat`, `y_test_cat`: Labels one-hot encodés

---

#### `compute_class_weights()`
Calcule les poids de classe pour gérer le déséquilibre.

```python
y_train = np.argmax(y_train_cat, axis=1)  # Convertir one-hot en integer

class_weights = compute_class_weights(
    y_train=y_train,
    categories=['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'],
    verbose=True
)
```

**Returns:** `{0: 0.95, 1: 1.05, 2: 1.2, 3: 0.8}` - Dict des poids

---

#### `create_data_generators()`
Crée les générateurs Keras avec augmentation optionnelle.

```python
train_gen, val_gen, test_gen = create_data_generators(
    X_train=X_train,
    y_train_cat=y_train_cat,
    X_val=X_val,
    y_val_cat=y_val_cat,
    X_test=X_test,           # Optionnel
    y_test_cat=y_test_cat,   # Optionnel
    batch_size=32,
    augment_train=True,  # Augmentation sur train uniquement
    verbose=True
)
```

**Returns:**
- `train_gen`: Générateur d'entraînement (avec augmentation si `augment_train=True`)
- `val_gen`: Générateur de validation (sans augmentation)
- `test_gen`: Générateur de test (sans augmentation, `None` si X_test non fourni)

**Augmentation appliquée (train uniquement):**
- Rotation: ±10°
- Shift: ±10%
- Zoom: ±10%
- Horizontal flip

---

### **2. Model Building**

#### `build_custom_cnn()`
Construit un CNN custom pour l'imagerie médicale.

```python
model = build_custom_cnn(
    input_shape=(128, 128, 3),
    num_classes=4,
    verbose=True
)
```

**Architecture:**
- 5 blocs convolutionnels: 32 → 64 → 128 → 256 → 512 filtres
- Batch Normalization après chaque Conv2D
- Dropout (0.25-0.5) pour régularisation
- L2 regularization sur couches denses
- ~15M paramètres

---

#### `compile_model()`
Compile le modèle avec métriques standards.

```python
model = compile_model(
    model=model,
    learning_rate=0.001,
    verbose=True
)
```

**Configuration:**
- Optimizer: Adam
- Loss: CategoricalCrossentropy
- Metrics: accuracy, auc, precision, recall

---

#### `create_callbacks()`
Crée les callbacks d'entraînement standards.

```python
callbacks = create_callbacks(
    models_dir=Path('results/models'),
    monitor='val_accuracy',
    patience_early_stop=15,
    patience_reduce_lr=5,
    verbose=True
)
```

**Callbacks inclus:**
1. `EarlyStopping` - Arrête si pas d'amélioration
2. `ReduceLROnPlateau` - Réduit le LR si plateau
3. `ModelCheckpoint` - Sauvegarde le meilleur modèle

---

### **3. Training & Evaluation**

#### `train_model()`
Entraîne le modèle.

```python
history = train_model(
    model=model,
    train_generator=train_gen,
    val_generator=val_gen,
    class_weights=class_weights,
    epochs=50,
    callbacks=callbacks,
    verbose=True
)
```

**Returns:** `keras.callbacks.History` - Historique d'entraînement

---

#### `evaluate_model()`
Évalue le modèle sur le test set.

```python
y_pred, y_pred_proba = evaluate_model(
    model=model,
    X_test=X_test,
    y_test_cat=y_test_cat,
    y_test=y_test,  # Labels integer
    categories=['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'],
    verbose=True
)
```

**Affiche:**
- Loss, Accuracy, AUC, Precision, Recall, F1-Score
- Classification report détaillé

**Returns:**
- `y_pred`: Prédictions (labels integer)
- `y_pred_proba`: Probabilités de prédiction

---

### **4. Visualization**

#### `plot_training_curves()`
Trace les courbes d'apprentissage.

```python
fig = plot_training_curves(
    history=history,
    save_path=Path('results/training_curves.png'),
    figsize=(15, 12)
)
plt.show()
```

**Graphiques:**
- Loss (train/val)
- Accuracy (train/val)
- AUC (train/val)
- Precision & Recall (train/val)

---

#### `plot_confusion_matrix()`
Trace la matrice de confusion.

```python
fig = plot_confusion_matrix(
    y_test=y_test,
    y_pred=y_pred,
    categories=['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'],
    save_path=Path('results/confusion_matrix.png'),
    figsize=(10, 8)
)
plt.show()
```

---

### **5. Interpretability**

#### `select_sample_images()`
Sélectionne des images échantillons pour l'analyse.

```python
sample_indices = select_sample_images(
    X_test=X_test,
    y_test=y_test,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    categories=['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'],
    n_samples=6,
    random_seed=42,
    verbose=True
)
```

---

#### `run_gradcam_analysis()`
Exécute l'analyse Grad-CAM.

```python
gradcam, heatmaps = run_gradcam_analysis(
    model=model,
    X_test=X_test,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    categories=['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'],
    sample_indices=sample_indices,
    save_dir=Path('results/interpretability'),
    verbose=True
)
```

**Returns:**
- `gradcam`: Instance de GradCAM
- `heatmaps`: Liste des heatmaps générées

---

## 📝 Exemple Complet

```python
# 1. Imports
from src.notebooks import *
import numpy as np
import matplotlib.pyplot as plt

# 2. Charger les données
image_paths, _, labels, labels_int = load_dataset(
    data_dir=config.data_dir,
    categories=config.classes
)

# 3. Preprocessing
pipeline = create_preprocessing_pipeline(img_size=(128, 128))
images = pipeline.fit_transform(image_paths)
images = images.astype('float32') / 255.0

# 4. Split
X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat = prepare_train_val_test_split(
    images, labels_int, num_classes=len(config.classes)
)

y_train = np.argmax(y_train_cat, axis=1)
y_test = np.argmax(y_test_cat, axis=1)

# 5. Class weights et generators
class_weights = compute_class_weights(y_train, config.classes)
train_gen, val_gen, test_gen = create_data_generators(
    X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, batch_size=32
)

# 6. Modèle
model = build_custom_cnn(input_shape=(128, 128, 3), num_classes=4)
model = compile_model(model)
callbacks = create_callbacks(models_dir=Path('results/models'))

# 7. Entraînement
history = train_model(model, train_gen, val_gen, class_weights, epochs=50, callbacks=callbacks)

# 8. Évaluation
y_pred, y_pred_proba = evaluate_model(model, X_test, y_test_cat, y_test, config.classes)

# 9. Visualisation
plot_training_curves(history, save_path=Path('results/curves.png'))
plot_confusion_matrix(y_test, y_pred, config.classes, save_path=Path('results/cm.png'))

# 10. Interprétabilité
samples = select_sample_images(X_test, y_test, y_pred, y_pred_proba, config.classes)
gradcam, heatmaps = run_gradcam_analysis(
    model, X_test, y_pred, y_pred_proba, config.classes, samples, Path('results')
)

plt.show()
```

---

## 🎯 Avantages

✅ **Code plus court** : Notebooks réduits de ~60%  
✅ **Réutilisable** : Mêmes fonctions dans tous les notebooks  
✅ **Maintenable** : Modifications centralisées  
✅ **Testé** : Fonctions unitaires testables  
✅ **Documenté** : Docstrings complètes  

---

## 🔄 Migration d'un Ancien Notebook

**Avant** (code répétitif):
```python
# 50+ lignes pour charger les données
image_paths = []
for cat in categories:
    # ...
    
# 30+ lignes pour la pipeline
# ...

# 60+ lignes pour le modèle
# ...
```

**Après** (fonctions réutilisables):
```python
image_paths, _, labels, labels_int = load_dataset(data_dir, categories)
pipeline = create_preprocessing_pipeline(img_size=(128, 128))
images = pipeline.fit_transform(image_paths)
model = build_custom_cnn(input_shape=(128, 128, 3))
```

**Réduction:** ~200 lignes → ~20 lignes ! 🚀

---

## 📚 Documentation Supplémentaire

Consultez les docstrings de chaque fonction pour plus de détails :

```python
help(load_dataset)
help(build_custom_cnn)
help(train_model)
```

---

## 🐛 Debugging

En cas d'erreur, vérifiez :

1. **Imports** : `from src.notebooks import *`
2. **Shapes** : Les dimensions des arrays
3. **Verbose** : Activer `verbose=True` pour diagnostiquer
4. **Logs** : Consulter les messages d'erreur détaillés

---

## 🚀 Prochaines Étapes

- [ ] Ajouter fonctions LIME/SHAP
- [ ] Support Transfer Learning (InceptionV3, ResNet)
- [ ] Export modèles (ONNX, TFLite)
- [ ] Comparaison automatique de modèles

---

**Auteur:** Data Pipeline Team  
**Date:** Novembre 2025  
**Licence:** MIT
