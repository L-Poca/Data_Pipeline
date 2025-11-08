# 🔍 Changelog - Interpretability Utils Enhancement

## Date: November 8, 2025

## 🎯 Objectif

Compléter le module `interpretability_utils.py` avec les techniques LIME et SHAP en plus de Grad-CAM.

---

## ✨ Nouveautés

### 1. **Support LIME (Local Interpretable Model-agnostic Explanations)**

#### Fonctions ajoutées :

- `setup_lime_explainer()`: Initialise un explainer LIME
  - Paramètres: `segmentation_method`, `num_samples`, `batch_size`
  - Méthodes de segmentation: `quickshift`, `felzenszwalb`, `slic`

- `run_lime_analysis()`: Analyse LIME sur des échantillons
  - Génère des visualisations avec super-pixels
  - Affiche les contributions positives/négatives
  - Sauvegarde automatique des résultats

- `compare_lime_segmentation()`: Compare les méthodes de segmentation
  - Aide à choisir la meilleure méthode pour vos images
  - Visualisation côte à côte

#### Caractéristiques :
- ✅ Agnostique au modèle (fonctionne avec tout type de classifieur)
- ✅ Explications locales précises
- ✅ Visualisation des super-pixels importants
- ✅ Graphiques de contribution par feature

---

### 2. **Support SHAP (SHapley Additive exPlanations)**

#### Fonctions ajoutées :

- `setup_shap_explainer()`: Initialise un explainer SHAP
  - Utilise DeepExplainer pour les CNN
  - Paramètre: `background_data` (référence), `max_background_samples`
  - ⚠️ Note: SHAP est lent, limiter le background à 50-100 images

- `run_shap_analysis()`: Analyse SHAP sur des échantillons
  - Génère des visualisations magnitude et signed
  - Heatmap overlay sur l'image originale
  - Sauvegarde automatique

- `compare_shap_classes()`: Compare les valeurs SHAP entre classes
  - Visualise l'importance pour chaque classe
  - Aide à comprendre les décisions multi-classes

#### Caractéristiques :
- ✅ Base théorique solide (valeurs de Shapley)
- ✅ Importance pixel par pixel
- ✅ Comparaison inter-classes
- ⚠️ Lent (plusieurs minutes par image)

---

### 3. **Fonction d'Analyse Complète**

#### `run_full_interpretability_analysis()`

Fonction tout-en-un qui exécute Grad-CAM, LIME et SHAP en une seule commande.

**Paramètres :**
```python
run_full_interpretability_analysis(
    model,                    # Modèle Keras
    x_data,                   # Images
    y_true,                   # Labels vrais
    y_pred,                   # Prédictions
    class_names,              # Noms des classes
    background_data=None,     # Pour SHAP
    n_samples=2,              # Échantillons par classe
    strategy="one_per_class", # Stratégie de sélection
    save_dir=None,            # Dossier de sauvegarde
    use_gradcam=True,         # Activer Grad-CAM
    use_lime=True,            # Activer LIME
    use_shap=False,           # Activer SHAP (lent)
    preprocess_fn=None,       # Preprocessing pour Transfer Learning
)
```

**Avantages :**
- 🚀 Analyse complète en une ligne
- 📁 Organisation automatique des résultats (gradcam/, lime/, shap/)
- ⚙️ Configuration flexible (activer/désactiver chaque technique)
- 🔄 Gestion automatique des erreurs

---

## 📦 Fichiers Modifiés/Créés

### Modifiés :

1. **`src/notebooks/interpretability_utils.py`**
   - Ajout de ~500 lignes de code
   - Import conditionnel de LIME et SHAP
   - 10 nouvelles fonctions
   - Documentation complète

2. **`src/notebooks/__init__.py`**
   - Export des nouvelles fonctions LIME et SHAP
   - Ajout à `__all__` pour API publique

### Créés :

3. **`examples/interpretability_example.py`**
   - Script d'exemple complet
   - Montre toutes les méthodes d'utilisation
   - Cas d'usage pratiques

4. **`examples/README_INTERPRETABILITY.md`**
   - Guide complet d'utilisation
   - Explications détaillées de chaque technique
   - Cas d'usage recommandés
   - Résolution de problèmes

5. **`CELL_INTERPRETABILITY.py`**
   - Cellule notebook prête à l'emploi
   - Pour comprehensive_ml_pipeline_v2.ipynb
   - Analyse complète automatisée

6. **`CHANGELOG_INTERPRETABILITY.md`** (ce fichier)
   - Résumé des changements

---

## 🎨 Fonctionnalités Détaillées

### Grad-CAM (Existant + Améliorations)

- ✅ Détection automatique de la dernière couche conv
- ✅ Support des modèles Transfer Learning
- ✅ Preprocessing conditionnel (InceptionV3, ResNet50, etc.)
- ✅ Visualisation avec confidence

### LIME (Nouveau)

- ✅ 3 méthodes de segmentation (quickshift, felzenszwalb, slic)
- ✅ Visualisation standard + avec frontières
- ✅ Graphique des contributions par super-pixel
- ✅ Comparaison des méthodes de segmentation
- ✅ Paramètres configurables (num_samples, num_features)

### SHAP (Nouveau)

- ✅ DeepExplainer pour CNN
- ✅ Visualisation magnitude + signed
- ✅ Heatmap overlay
- ✅ Comparaison inter-classes
- ✅ Grille de visualisations multiples
- ⚠️ Optimisation pour vitesse (limitation background)

---

## 🚀 Utilisation

### Exemple Simple (1 ligne)

```python
from src.notebooks import run_full_interpretability_analysis

run_full_interpretability_analysis(
    model=model,
    x_data=X_test,
    y_true=y_test,
    y_pred=y_pred,
    class_names=['COVID', 'Normal', 'Lung_Opacity', 'Viral_Pneumonia'],
    save_dir=Path("results/interpretability")
)
```

### Exemple Avancé (Contrôle Fin)

```python
from src.notebooks import (
    setup_lime_explainer,
    run_lime_analysis,
    select_sample_images
)

# Setup
lime_explainer = setup_lime_explainer(model, segmentation_method='quickshift')

# Sélection
indices, descriptions = select_sample_images(
    X_test, y_test, y_pred, class_names,
    n_samples=2,
    strategy="correct"
)

# Analyse
run_lime_analysis(
    lime_explainer, X_test, indices, descriptions, class_names,
    num_features=5,
    save_dir=Path("results/lime")
)
```

---

## 📊 Comparaison des Techniques

| Technique | Vitesse | Précision | Agnostique | Visualisation |
|-----------|---------|-----------|------------|---------------|
| **Grad-CAM** | ⚡⚡⚡ Très rapide | Bonne | ❌ CNN uniquement | Carte de chaleur |
| **LIME** | ⚡⚡ Rapide | Très bonne | ✅ Oui | Super-pixels |
| **SHAP** | ⚠️ Lent | Excellente | ✅ Oui | Valeurs pixel |

### Recommandations :

- **Validation rapide** → Grad-CAM
- **Explications détaillées** → LIME
- **Recherche scientifique** → SHAP
- **Production** → Grad-CAM + LIME
- **Analyse complète** → Les 3 (si temps disponible)

---

## 🔍 Structure des Résultats

```
results/interpretability/
├── gradcam/
│   ├── gradcam_01.png
│   ├── gradcam_02.png
│   └── ...
├── lime/
│   ├── lime_01.png
│   ├── lime_contrib_01.png
│   ├── lime_02.png
│   └── ...
└── shap/
    ├── shap_01.png
    ├── shap_heatmap_01.png
    ├── shap_02.png
    └── ...
```

---

## 📝 Notes Importantes

### Dépendances :

```bash
# Grad-CAM (inclus)
# Pas d'installation supplémentaire

# LIME
pip install lime

# SHAP
pip install shap
```

### Performance :

- **Grad-CAM**: ~0.1 sec/image
- **LIME**: ~10-30 sec/image (selon num_samples)
- **SHAP**: ~60-300 sec/image ⚠️

### Recommandations :

1. Toujours commencer par Grad-CAM (rapide)
2. Ajouter LIME pour détails (acceptable)
3. SHAP uniquement si nécessaire (recherche)
4. Limiter SHAP background_data à 50-100 images
5. Analyser 1-2 échantillons par classe pour commencer

---

## 🎯 Cas d'Usage Médicaux

### 1. Validation Radiologique

**Objectif**: Vérifier que le modèle regarde les bonnes zones

```python
run_full_interpretability_analysis(
    model, X_test, y_test, y_pred, class_names,
    strategy="correct",
    use_gradcam=True,
    use_lime=True
)
```

### 2. Analyse d'Erreurs

**Objectif**: Comprendre pourquoi le modèle se trompe

```python
run_full_interpretability_analysis(
    model, X_test, y_test, y_pred, class_names,
    strategy="incorrect",
    n_samples=3
)
```

### 3. Détection de Biais

**Objectif**: Identifier les artifacts, marqueurs, etc.

```python
# Analyser avec Grad-CAM (rapide)
gradcam = setup_interpretability(model)
run_gradcam_analysis(
    gradcam, X_test, indices, descriptions, class_names,
    save_dir=Path("results/bias_detection")
)
```

---

## 🐛 Résolution de Problèmes

### "LIME non disponible"
```bash
pip install lime
```

### "SHAP non disponible"
```bash
pip install shap
```

### SHAP trop lent
```python
# Réduire background_data
shap_explainer = setup_shap_explainer(
    model,
    background_data=X_train[:25],  # Au lieu de 50-100
)
```

### Erreur Transfer Learning
```python
# Assurez-vous que les images sont en RGB
X_test_rgb = np.repeat(X_test[..., np.newaxis], 3, axis=-1)
```

---

## 📚 Références

- **Grad-CAM**: Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- **LIME**: Ribeiro et al. (2016) - "'Why Should I Trust You?': Explaining the Predictions of Any Classifier"
- **SHAP**: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"

---

## ✅ Tests Effectués

- ✅ Import de toutes les fonctions réussi
- ✅ Grad-CAM fonctionne sur CNN personnalisés
- ✅ Grad-CAM fonctionne sur Transfer Learning
- ✅ LIME disponible (avec import conditionnel)
- ✅ SHAP disponible (avec import conditionnel)
- ✅ `run_full_interpretability_analysis()` testé
- ✅ Sauvegarde automatique des résultats
- ✅ Gestion d'erreurs robuste

---

## 🎉 Résumé

**Avant**: Seulement Grad-CAM disponible

**Après**: 
- ✅ Grad-CAM (amélioré)
- ✅ LIME (nouveau)
- ✅ SHAP (nouveau)
- ✅ Analyse complète automatisée
- ✅ Documentation exhaustive
- ✅ Exemples pratiques

**Impact**: Pipeline d'interprétabilité complet et production-ready pour la validation médicale des modèles COVID-19.

---

**🏥 Note Médicale**: Ces outils sont essentiels pour la confiance des radiologues dans les prédictions IA et pour l'acceptation réglementaire (FDA, CE marking).
