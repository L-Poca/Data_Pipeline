# 🔍 Guide d'Interprétabilité - Grad-CAM, LIME & SHAP

## 📋 Vue d'ensemble

Le module `interpretability_utils` fournit des fonctions pour expliquer les prédictions des modèles de Deep Learning. Trois techniques principales sont disponibles :

1. **Grad-CAM** (Gradient-weighted Class Activation Mapping)
2. **LIME** (Local Interpretable Model-agnostic Explanations)
3. **SHAP** (SHapley Additive exPlanations)

---

## 🚀 Installation

```bash
# Grad-CAM (inclus dans le projet)
# Pas d'installation supplémentaire requise

# LIME
pip install lime

# SHAP (optionnel, lent)
pip install shap
```

---

## 📚 Guide d'utilisation

### Méthode 1 : Analyse Complète (Recommandé)

La façon la plus simple d'utiliser toutes les techniques :

```python
from src.notebooks import run_full_interpretability_analysis

# Lancer l'analyse complète
run_full_interpretability_analysis(
    model=model,
    x_data=X_test,
    y_true=y_test,
    y_pred=y_pred,
    class_names=['COVID', 'Normal', 'Lung_Opacity', 'Viral_Pneumonia'],
    background_data=X_train[:100],  # Pour SHAP
    n_samples=2,  # Échantillons par classe
    strategy="one_per_class",
    save_dir=Path("results/interpretability"),
    use_gradcam=True,
    use_lime=True,
    use_shap=False,  # SHAP est lent
)
```

**Paramètres :**
- `strategy`: `'one_per_class'`, `'correct'`, `'incorrect'`, `'random'`
- `n_samples`: Nombre d'échantillons à analyser
- `use_gradcam`, `use_lime`, `use_shap`: Activer/désactiver chaque technique

---

### Méthode 2 : Analyses Séparées

Pour un contrôle plus fin :

#### 🔥 Grad-CAM

```python
from src.notebooks import (
    setup_interpretability,
    run_gradcam_analysis,
    select_sample_images,
    get_preprocessing_function
)

# 1. Setup
gradcam = setup_interpretability(model, verbose=True)

# 2. Sélectionner des échantillons
indices, descriptions = select_sample_images(
    X_test, y_test, y_pred, class_names,
    n_samples=2,
    strategy="correct"
)

# 3. Fonction de preprocessing (pour Transfer Learning)
preprocess_fn = get_preprocessing_function('InceptionV3')

# 4. Analyser
run_gradcam_analysis(
    gradcam, X_test, indices, descriptions, class_names,
    y_pred_probs=y_pred_probs,
    save_dir=Path("results/gradcam"),
    preprocess_fn=preprocess_fn
)
```

**Avantages Grad-CAM :**
- ✅ Rapide
- ✅ Visualisation intuitive
- ✅ Fonctionne avec tous les CNN
- ✅ Montre les régions d'attention

---

#### 🧪 LIME

```python
from src.notebooks import (
    setup_lime_explainer,
    run_lime_analysis,
    compare_lime_segmentation
)

# 1. Setup
lime_explainer = setup_lime_explainer(
    model,
    segmentation_method='quickshift',  # 'quickshift', 'felzenszwalb', 'slic'
    num_samples=1000,
    verbose=True
)

# 2. Analyser
run_lime_analysis(
    lime_explainer, X_test, indices, descriptions, class_names,
    y_pred=y_pred,
    num_features=5,  # Nombre de super-pixels
    save_dir=Path("results/lime")
)

# 3. Comparer les méthodes de segmentation
compare_lime_segmentation(
    lime_explainer,
    X_test[0],
    save_dir=Path("results/lime")
)
```

**Avantages LIME :**
- ✅ Agnostique au modèle
- ✅ Explications locales précises
- ✅ Montre les super-pixels importants
- ✅ Contributions positives/négatives

**Méthodes de segmentation :**
- `quickshift`: Rapide, bonne qualité générale (par défaut)
- `felzenszwalb`: Segments plus grands
- `slic`: Segments réguliers (grille)

---

#### 📊 SHAP

```python
from src.notebooks import (
    setup_shap_explainer,
    run_shap_analysis,
    compare_shap_classes
)

# 1. Setup (LENT - limiter background_data)
shap_explainer = setup_shap_explainer(
    model,
    background_data=X_train[:50],  # 50-100 échantillons max
    max_background_samples=50,
    verbose=True
)

# 2. Analyser
run_shap_analysis(
    shap_explainer, X_test, indices[:2], descriptions[:2], class_names,
    y_pred=y_pred,
    save_dir=Path("results/shap")
)

# 3. Comparer les classes
compare_shap_classes(
    shap_explainer,
    X_test[0],
    class_names,
    save_dir=Path("results/shap")
)
```

**Avantages SHAP :**
- ✅ Théorie mathématique solide
- ✅ Importance pixel par pixel
- ✅ Comparaison inter-classes
- ⚠️ **Lent** (plusieurs minutes par image)

---

## 🎯 Cas d'usage

### Cas 1 : Validation Médicale

**Objectif :** Vérifier que le modèle se concentre sur les bonnes régions pulmonaires.

```python
# Analyser les prédictions correctes
run_full_interpretability_analysis(
    model=model,
    x_data=X_test,
    y_true=y_test,
    y_pred=y_pred,
    class_names=class_names,
    strategy="correct",
    n_samples=5,
    use_gradcam=True,
    use_lime=True,
    save_dir=Path("results/validation")
)
```

---

### Cas 2 : Analyse d'Erreurs

**Objectif :** Comprendre pourquoi le modèle se trompe.

```python
# Analyser les erreurs
run_full_interpretability_analysis(
    model=model,
    x_data=X_test,
    y_true=y_test,
    y_pred=y_pred,
    class_names=class_names,
    strategy="incorrect",
    n_samples=3,
    use_gradcam=True,
    use_lime=True,
    save_dir=Path("results/error_analysis")
)
```

---

### Cas 3 : Comparaison de Modèles

**Objectif :** Comparer l'interprétabilité de différents modèles.

```python
models = {
    'InceptionV3': model_inception,
    'ResNet50': model_resnet,
    'VGG16': model_vgg
}

for name, model in models.items():
    run_full_interpretability_analysis(
        model=model,
        x_data=X_test,
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        strategy="one_per_class",
        save_dir=Path(f"results/comparison/{name}"),
        use_gradcam=True,
        use_lime=False
    )
```

---

## 🔬 Détails Techniques

### Grad-CAM

**Principe :**
- Utilise les gradients de la dernière couche convolutionnelle
- Génère une carte de chaleur des régions importantes
- Rapide et efficace

**Quand l'utiliser :**
- Validation rapide des prédictions
- Visualisation pour présentation
- Détection de biais (artifacts, marqueurs)

**Limitations :**
- Seulement pour les CNN
- Résolution limitée par la dernière couche conv

---

### LIME

**Principe :**
- Perturbe l'image en masquant des super-pixels
- Entraîne un modèle linéaire local
- Identifie les régions les plus influentes

**Quand l'utiliser :**
- Explications détaillées pixel par pixel
- Agnostique au modèle (fonctionne avec tout)
- Contributions positives vs négatives

**Paramètres importants :**
- `num_samples`: Plus = meilleur mais plus lent (500-1000 recommandé)
- `num_features`: Nombre de super-pixels à afficher (3-10)
- `segmentation_method`: Qualité de la segmentation

---

### SHAP

**Principe :**
- Utilise la théorie des jeux (valeurs de Shapley)
- Calcule l'importance de chaque pixel
- Garanties théoriques d'additivité

**Quand l'utiliser :**
- Recherche scientifique (rigueur mathématique)
- Importance globale des features
- Comparaison inter-classes

**⚠️ Avertissements :**
- **Très lent** (plusieurs minutes par image)
- Limiter `background_data` à 50-100 images
- Pas pour analyse en temps réel

---

## 📊 Interprétation des Résultats

### Grad-CAM : Carte de Chaleur

- **Rouge/Jaune** : Régions très importantes
- **Bleu/Vert** : Régions moins importantes
- **Zones concentrées** : Bonne confiance
- **Zones dispersées** : Incertitude

**Ce qu'il faut vérifier :**
- Le modèle regarde-t-il les poumons ou autre chose ?
- Les zones d'attention correspondent-elles aux symptômes ?
- Y a-t-il des artifacts (coins, bordures, marqueurs) ?

---

### LIME : Super-pixels

- **Vert** : Contribution positive à la classe
- **Rouge** : Contribution négative
- **Plus la zone est grande** : Plus l'influence est forte

**Ce qu'il faut vérifier :**
- Les super-pixels importants sont-ils médicalement pertinents ?
- Les contributions négatives font-elles sens ?
- La segmentation capture-t-elle bien les structures ?

---

### SHAP : Valeurs de Shapley

- **Magnitude** : Importance absolue (heatmap rouge)
- **Signed** : Contribution positive (rouge) vs négative (bleu)
- **Plus intense** : Plus important pour la décision

**Ce qu'il faut vérifier :**
- Cohérence avec Grad-CAM et LIME ?
- Distribution des valeurs SHAP
- Comparaison entre classes

---

## 🎨 Personnalisation

### Changer le nombre d'échantillons

```python
# Plus d'échantillons par classe
run_full_interpretability_analysis(
    ...,
    n_samples=5,  # Au lieu de 2
    strategy="correct"
)
```

### Désactiver certaines techniques

```python
# Seulement Grad-CAM (rapide)
run_full_interpretability_analysis(
    ...,
    use_gradcam=True,
    use_lime=False,
    use_shap=False
)
```

### Analyser des échantillons spécifiques

```python
# Choisir manuellement les indices
indices = [10, 25, 42, 100]
descriptions = ["Échantillon 1", "Échantillon 2", ...]

run_gradcam_analysis(
    gradcam, X_test, indices, descriptions, class_names,
    save_dir=Path("results/custom")
)
```

---

## 📁 Structure des Résultats

```
results/interpretability/
├── gradcam/
│   ├── gradcam_01.png
│   ├── gradcam_02.png
│   └── ...
├── lime/
│   ├── lime_01.png
│   ├── lime_contrib_01.png
│   └── ...
└── shap/
    ├── shap_01.png
    ├── shap_heatmap_01.png
    └── ...
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

### "Aucune couche convolutionnelle trouvée"

Pour Grad-CAM, le modèle doit avoir des couches convolutionnelles.

### SHAP trop lent

Réduire le nombre d'échantillons background :

```python
shap_explainer = setup_shap_explainer(
    model,
    background_data=X_train[:25],  # Seulement 25 au lieu de 100
)
```

### Erreur de dimension Transfer Learning

Assurez-vous que les images sont en RGB (3 canaux) :

```python
X_test_rgb = np.repeat(X_test[..., np.newaxis], 3, axis=-1)
```

---

## 📚 Références

- **Grad-CAM**: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks" (2017)
- **LIME**: Ribeiro et al. "'Why Should I Trust You?': Explaining Predictions" (2016)
- **SHAP**: Lundberg & Lee "A Unified Approach to Interpreting Model Predictions" (2017)

---

## 🤝 Exemple Complet

Voir `examples/interpretability_example.py` pour un exemple complet d'utilisation.

```bash
python examples/interpretability_example.py
```

---

## 💡 Conseils

1. **Commencez par Grad-CAM** (rapide, intuitif)
2. **Ajoutez LIME** pour plus de détails
3. **Utilisez SHAP** seulement si nécessaire (recherche)
4. **Analysez plusieurs échantillons** par classe
5. **Comparez correctes vs incorrectes** pour comprendre les erreurs
6. **Vérifiez les biais** (artifacts, marqueurs médicaux)
7. **Documentez vos findings** pour validation médicale

---

**📝 Note :** Pour une utilisation en milieu clinique, toujours faire valider les explications par un radiologue qualifié.
