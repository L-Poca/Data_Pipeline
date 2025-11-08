# 🦠 Détection COVID-19
## Application de détection COVID-19 à partir d'images radiographiques

## Installation

Ce projet est maintenant un package Python installable. Pour l'installer en mode développement :

```bash
pip install -e .
```

Pour plus d'informations sur l'installation, consultez [INSTALLATION.md](INSTALLATION.md).

## 🌟 Fonctionnalités Clés

### 📊 Template d'Analyse Exploratoire Avancée (NEW!)

Un notebook template complet pour l'analyse exploratoire des données médicales :
- **Réduction de dimensionnalité** : PCA, t-SNE avec visualisations 2D/3D
- **Clustering avancé** : K-Means, DBSCAN, Hierarchical avec métriques de qualité
- **Tests statistiques** : ANOVA, Kruskal-Wallis pour comparaison de classes
- **Features texturales** : Extraction GLCM (contrast, homogeneity, energy, etc.)
- **Rapport HTML automatique** : Génération d'un rapport professionnel interactif

👉 **[Voir notebooks/advanced_eda_template.ipynb](notebooks/advanced_eda_template.ipynb)**  
📖 **[Documentation complète](notebooks/README_advanced_eda.md)**

### 🔧 Configuration Unifiée

- Configuration centralisée via JSON (voir `config/`)
- Auto-détection d'environnement (Colab, WSL, Local)
- `CELL_CONFIG_STANDALONE.py` pour notebooks autonomes

### 🧪 Pipelines de Transformation

- Transformateurs sklearn-compatibles pour images médicales
- Preprocessing, augmentation, extraction de features
- Pipeline modulaire et réutilisable

## Usage

### Quick Start avec Notebooks

Pour démarrer rapidement avec un notebook Jupyter :

1. **Configuration automatique** : Copiez le contenu de `CELL_CONFIG_STANDALONE.py` dans la première cellule
2. **Analyse exploratoire** : Utilisez le template `notebooks/advanced_eda_template.ipynb`
3. **Documentation** : Consultez `notebooks/README_advanced_eda.md`

### Usage Programmatique

Après installation, vous pouvez importer les transformateurs :

```python
from src.features import (
    ImageLoader,
    ImageResizer,
    ImageNormalizer,
    # ... et d'autres transformateurs
)
```

Voir [examples/](examples/) pour des exemples d'utilisation détaillés.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py