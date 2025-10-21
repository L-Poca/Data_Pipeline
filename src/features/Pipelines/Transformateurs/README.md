# Transformateurs - Structure Réorganisée

## ⚠️ Nouvelle Organisation

Les transformateurs ont été réorganisés dans une structure hiérarchique plus claire :

```
src/features/
├── __init__.py                 # Point d'entrée principal, importe depuis Pipelines
├── Pipelines/
│   ├── __init__.py            # Importe depuis Transformateurs
│   └── Transformateurs/
│       ├── __init__.py        # Expose tous les transformateurs
│       ├── image_loaders.py   # ImageLoader
│       ├── image_preprocessing.py  # ImageResizer, ImageNormalizer, ImageMasker, ImageFlattener, ImageBinarizer
│       ├── image_augmentation.py   # ImageAugmenter, ImageRandomCropper
│       ├── image_features.py       # ImageHistogram, ImagePCA, ImageStandardScaler
│       ├── utilities.py            # VisualizeTransformer, SaveTransformer
│       └── README.md               # Cette documentation
└── [autres modules...]
```

## 📥 Image Loaders (`image_loaders.py`)
- **ImageLoader** : Charge les images à partir de chemins de fichiers

## 🔧 Image Preprocessing (`image_preprocessing.py`)
- **ImageResizer** : Redimensionne les images
- **ImageNormalizer** : Normalise les pixels entre 0 et 1
- **ImageMasker** : Applique des masques sur les images
- **ImageFlattener** : Aplatit les images pour les modèles ML
- **ImageBinarizer** : Binarise les images avec un seuil

## 🔄 Image Augmentation (`image_augmentation.py`)
- **ImageAugmenter** : Applique des transformations d'augmentation (flip horizontal)
- **ImageRandomCropper** : Effectue un crop aléatoire

## 📊 Image Features (`image_features.py`)
- **ImageHistogram** : Calcule les histogrammes d'intensité
- **ImagePCA** : Réduction de dimension par ACP
- **ImageStandardScaler** : Standardisation des features

## 🛠️ Utilities (`utilities.py`)
- **VisualizeTransformer** : Visualisation des échantillons
- **SaveTransformer** : Sauvegarde des features extraites

## ✅ Utilisation (Inchangée)

```python
# L'import reste identique grâce à la structure des __init__.py
from src.features import (
    ImageLoader,
    ImageResizer,
    ImageNormalizer,
    ImageFlattener,
    # ... tous les autres transformateurs
)
from sklearn.pipeline import Pipeline

# Créer un pipeline
pipeline = Pipeline([
    ('loader', ImageLoader()),
    ('resizer', ImageResizer(img_size=(256, 256))),
    ('normalizer', ImageNormalizer()),
    ('flattener', ImageFlattener()),
])

# Utiliser le pipeline
X_transformed = pipeline.fit_transform(image_paths)
```

## 🔄 Migration

- **✅ Aucun changement** requis dans le code utilisateur
- **✅ Les imports** `from src.features import ...` fonctionnent toujours
- **✅ Structure hiérarchique** améliore l'organisation
- **✅ Compatibilité** scikit-learn préservée

## 🏗️ Compatibilité

Tous les transformateurs héritent de `sklearn.base.BaseEstimator` et `sklearn.base.TransformerMixin`, ce qui les rend compatibles avec les pipelines scikit-learn.
