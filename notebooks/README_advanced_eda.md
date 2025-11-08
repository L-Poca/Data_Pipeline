# 🦠 Advanced EDA Template - COVID-19 Radiography Dataset

## 📋 Overview

Ce notebook template fournit une analyse exploratoire avancée (EDA) complète pour le dataset COVID-19 Radiography, incluant:

- ✅ Analyse statistique détaillée
- ✅ Réduction de dimensionnalité (PCA, t-SNE)
- ✅ Clustering avancé (K-Means, DBSCAN, Hierarchical)
- ✅ Tests statistiques (ANOVA, Kruskal-Wallis)
- ✅ Extraction de features texturales (GLCM)
- ✅ Génération automatique de rapport HTML

---

## 🚀 Quick Start

### 1. Prérequis

```bash
# Installer le package en mode développement
pip install -e .
```

### 2. Lancer le Notebook

```bash
# Depuis le répertoire racine
jupyter notebook notebooks/advanced_eda_template.ipynb
```

ou depuis Google Colab:
- Téléverser le notebook
- Exécuter la première cellule (CELL_CONFIG_STANDALONE)
- Le reste s'exécute automatiquement

---

## 📊 Contenu du Notebook

### Cell 1: Configuration (CELL_CONFIG_STANDALONE)
- ✅ Détection automatique de l'environnement (Colab, WSL, Local)
- ✅ Installation et configuration automatique
- ✅ Imports de tous les transformateurs nécessaires
- ✅ Configuration des chemins et paramètres

### Section 1: Chargement des Données
- Chargement des images radiographiques
- Chargement des masques de segmentation
- Preprocessing avec pipelines réutilisables
- Normalisation des images

### Section 2: Analyse Statistique de Base
- Distribution des classes (histogrammes, pie charts)
- Statistiques descriptives par classe (mean, std, min, max, quartiles)
- Box plots des intensités moyennes
- Tests de normalité

### Section 3: PCA (Principal Component Analysis)
- Réduction de dimensionnalité
- Analyse de la variance expliquée
- Visualisations 2D et 3D des composantes principales
- Identification des features importantes

### Section 4: t-SNE
- Visualisation non-linéaire haute dimension
- Projection 2D avec différents perplexity
- Comparaison avec PCA
- Visualisation interactive avec Plotly (optionnel)

### Section 5: Clustering
#### K-Means
- Optimisation du nombre de clusters (Elbow method)
- Métriques de qualité: Silhouette, Davies-Bouldin, Calinski-Harabasz
- Comparaison clusters vs vraies classes

#### DBSCAN
- Clustering basé sur la densité
- Détection automatique de bruit
- Identification des outliers

#### Hierarchical Clustering
- Dendrogram pour visualiser la hiérarchie
- Agglomerative clustering
- Comparaison des méthodes de linkage

### Section 6: Tests Statistiques & Features
#### Tests Statistiques
- **ANOVA**: Test de différence entre moyennes des classes
- **Kruskal-Wallis**: Alternative non-paramétrique
- Interprétation des p-values

#### Features Texturales (GLCM)
- Contrast: Mesure du contraste local
- Dissimilarity: Variation de niveaux de gris
- Homogeneity: Uniformité de l'image
- Energy: Somme des carrés des éléments
- Correlation: Corrélation linéaire entre pixels
- Entropy: Entropie de Shannon

### Section 7: Génération Rapport HTML
- Rapport professionnel auto-généré
- Toutes les métriques consolidées
- Export JSON pour usage programmatique
- Design responsive et moderne

---

## 🎯 Utilisation Avancée

### Personnaliser les Paramètres

Modifiez le fichier `config/default_config.json`:

```json
{
  "memory": {
    "max_images_per_class": 1000,  // Limiter le nombre d'images chargées
    "sample_size_analysis": 200     // Échantillon pour features texturales
  },
  "transformers": {
    "pca": {
      "n_components": 50,            // Nombre de composantes PCA
      "whiten": false
    }
  }
}
```

### Adapter à un Nouveau Dataset

1. **Modifier le chemin des données**:
```python
# Dans la section de chargement
data_dir = Path('/path/to/your/dataset')
categories = ['Class1', 'Class2', 'Class3']
```

2. **Ajuster le preprocessing**:
```python
# Modifier la taille des images
pipeline_img = create_preprocessing_pipeline(
    img_size=(224, 224),  # Nouvelle taille
    color_mode='RGB'      # ou 'L' pour grayscale
)
```

3. **Personnaliser les analyses**:
```python
# Modifier les paramètres de clustering
kmeans = KMeans(n_clusters=5)  # Ajuster le nombre de clusters
dbscan = DBSCAN(eps=2.0, min_samples=10)  # Ajuster les paramètres
```

---

## 📈 Interprétation des Résultats

### Métriques de Clustering

#### Silhouette Score
- **Range**: [-1, 1]
- **Interprétation**: 
  - > 0.7: Excellent clustering
  - 0.5-0.7: Bon clustering
  - 0.25-0.5: Clustering faible
  - < 0.25: Mauvais clustering

#### Davies-Bouldin Index
- **Range**: [0, ∞]
- **Interprétation**: Plus bas = meilleur
- < 1: Excellent clustering

#### Calinski-Harabasz Score
- **Range**: [0, ∞]
- **Interprétation**: Plus élevé = meilleur

### Tests Statistiques

#### p-value < 0.05
✅ Différence statistiquement significative entre les classes

#### p-value ≥ 0.05
❌ Pas de différence significative détectée

---

## 🔧 Dépendances

### Packages Requis

```txt
# Core
numpy>=2.0.0
pandas>=2.2.0
scipy>=1.14.0

# Machine Learning
scikit-learn>=1.5.0
scikit-image>=0.24.0

# Deep Learning
tensorflow>=2.18.0

# Visualization
matplotlib>=3.9.0
seaborn>=0.13.2
plotly>=5.0.0 (optionnel)

# Utils
tqdm>=4.67.0
```

---

## 📝 Exemples de Sorties

### Rapport HTML
Le rapport HTML généré contient:
- ✅ Statistiques du dataset
- ✅ Résultats PCA et variance expliquée
- ✅ Métriques de clustering pour chaque méthode
- ✅ Résultats des tests statistiques
- ✅ Features texturales par classe
- ✅ Design professionnel et responsive

### JSON Summary
```json
{
  "generation_date": "2025-11-08 12:00:00",
  "dataset": {
    "n_images": 4000,
    "n_classes": 4,
    "class_distribution": {...}
  },
  "pca": {
    "total_variance_explained": 0.95,
    ...
  },
  "clustering": {...},
  "statistical_tests": {...}
}
```

---

## 🚨 Troubleshooting

### Erreur: Memory Error
```python
# Réduire le nombre d'images chargées
n_images_per_class=500  # Au lieu de None (toutes)
```

### Erreur: t-SNE trop lent
```python
# Utiliser moins de composantes PCA avant t-SNE
n_components = 30  # Au lieu de 50
```

### Erreur: GLCM features lent
```python
# Réduire le nombre d'échantillons
n_samples_features = 50  # Au lieu de 100
```

---

## 🎓 Ressources

### Documentation
- [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [scikit-learn t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [scikit-image GLCM](https://scikit-image.org/docs/stable/api/skimage.feature.html#graycomatrix)

### Papers
- van der Maaten & Hinton (2008). "Visualizing Data using t-SNE"
- Haralick et al. (1973). "Textural Features for Image Classification"

---

## 👥 Contribution

Pour contribuer au template:

1. Fork le repository
2. Créer une branche: `git checkout -b feature/improvement`
3. Commit les changements: `git commit -m 'Add feature'`
4. Push: `git push origin feature/improvement`
5. Créer une Pull Request

---

## 📄 Licence

MIT License - voir LICENSE file

---

## 📧 Contact

Data Pipeline Team - L-Poca/Data_Pipeline

**Auteurs:** Rafael Cepa, Cirine, Steven Moire  
**Date:** Novembre 2025  
**Version:** 1.0

---

## ⭐ Features à Venir

- [ ] Support pour d'autres formats d'images (DICOM, NIFTI)
- [ ] Analyse temporelle pour séries d'images
- [ ] Intégration UMAP pour réduction de dimensionnalité
- [ ] Export rapport PDF en plus de HTML
- [ ] Dashboard interactif Streamlit
- [ ] Analyse comparative multi-datasets
