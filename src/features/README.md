# Features Module

Ce module contient tous les transformateurs et estimateurs personnalisés pour le pipeline de données COVID-19.

## Organisation

Les transformateurs ont été extraits du notebook Jupyter et organisés en modules logiques :

### 📥 Image Loaders (`image_loaders.py`)
- **ImageLoader** : Charge les images à partir de chemins de fichiers

### 🔧 Image Preprocessing (`image_preprocessing.py`)
- **ImageResizer** : Redimensionne les images
- **ImageNormalizer** : Normalise les pixels entre 0 et 1
- **ImageMasker** : Applique des masques sur les images
- **ImageFlattener** : Aplatit les images pour les modèles ML
- **ImageBinarizer** : Binarise les images avec un seuil

### 🔄 Image Augmentation (`image_augmentation.py`)
- **ImageAugmenter** : Applique des transformations d'augmentation (flip horizontal)
- **ImageRandomCropper** : Effectue un crop aléatoire

### 📊 Image Features (`image_features.py`)
- **ImageHistogram** : Calcule les histogrammes d'intensité
- **ImagePCA** : Réduction de dimension par ACP
- **ImageStandardScaler** : Standardisation des features

### 🛠️ Utilities (`utilities.py`)
- **VisualizeTransformer** : Visualisation des échantillons
- **SaveTransformer** : Sauvegarde des features extraites

## Utilisation

```python
from src.features import (
    ImageLoader,
    ImageResizer,
    ImageNormalizer,
    ImageFlattener,
)
from sklearn.pipeline import Pipeline

# Créer un pipeline
pipeline = Pipeline([
    ('loader', ImageLoader(img_size=(128, 128))),
    ('resizer', ImageResizer(img_size=(256, 256))),
    ('normalizer', ImageNormalizer()),
    ('flattener', ImageFlattener()),
])

# Utiliser le pipeline
X_transformed = pipeline.fit_transform(image_paths)
```

## Compatibilité

Tous les transformateurs héritent de `sklearn.base.BaseEstimator` et `sklearn.base.TransformerMixin`, ce qui les rend compatibles avec les pipelines scikit-learn.
