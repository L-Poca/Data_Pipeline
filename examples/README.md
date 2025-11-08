# 📚 Examples Directory - Data Pipeline

Ce dossier contient des exemples et de la documentation pour bien utiliser le Data Pipeline.

## 🎯 Fichiers principaux

### 1. Configuration (`config`)

#### 📖 `CONFIG_USAGE_GUIDE.md`
Guide complet d'utilisation de l'objet `config`:
- Vue d'ensemble de la structure
- Exemples pour chaque cas d'usage
- Bonnes pratiques DO/DON'T
- Guide de migration depuis ancien code
- FAQ

**👉 Commencez par ici pour comprendre `config`!**

#### 💻 `config_usage_examples.py`
Exemples de code concrets et fonctionnels:
- Accès aux chemins
- Paramètres d'entraînement
- Construction de pipelines
- Augmentation de données
- Visualisation
- Interprétabilité (Grad-CAM, SHAP)
- Modèles ML (Random Forest, XGBoost)
- Pipeline complet d'entraînement

**👉 Copiez-collez ces snippets dans vos notebooks!**

#### 🔄 `demo_config_benefits.py`
Démonstration comparative AVANT/APRÈS:
- Code ancien vs nouveau
- Tableau comparatif des approches
- Exemples concrets de gains
- Guide d'utilisation pas à pas

**👉 Lancez-le pour voir les améliorations: `python3 demo_config_benefits.py`**

### 2. Usage basique

#### 🚀 `basic_usage.py`
Exemple d'usage simple du pipeline (version legacy).

**Note**: Pour les nouveaux projets, utilisez plutôt `CELL_CONFIG_STANDALONE.py` et les exemples de `config_usage_examples.py`.

### 3. Advanced EDA Template

#### 📊 `advanced_eda_usage.py`
Exemple d'utilisation du notebook template d'analyse exploratoire avancée:
- Workflow complet de l'EDA
- Exemples de chaque étape d'analyse
- Guide conceptuel des fonctionnalités

**👉 Voir aussi: `notebooks/advanced_eda_template.ipynb` et `notebooks/README_advanced_eda.md`**

## 📂 Structure

```
examples/
├── README.md                      # ← Vous êtes ici
├── CONFIG_USAGE_GUIDE.md          # Documentation complète de config
├── config_usage_examples.py       # Exemples de code pratiques
├── demo_config_benefits.py        # Démonstration comparative
├── advanced_eda_usage.py          # Exemple d'usage du template EDA avancé
└── basic_usage.py                 # Usage basique (legacy)
```

## 🎓 Guide d'apprentissage recommandé

### Pour débuter:
1. Lisez `CONFIG_USAGE_GUIDE.md` (10 min)
2. Lancez `demo_config_benefits.py` (2 min)
3. Parcourez `config_usage_examples.py` (15 min)

### Pour la pratique:
1. Copiez `CELL_CONFIG_STANDALONE.py` dans votre notebook
2. Utilisez les exemples de `config_usage_examples.py`
3. Référez-vous à `CONFIG_USAGE_GUIDE.md` en cas de doute

## 💡 Cas d'usage typiques

### Je veux...

#### ...démarrer un nouveau notebook
→ Copiez `CELL_CONFIG_STANDALONE.py` en première cellule

#### ...comprendre comment utiliser config
→ Lisez `CONFIG_USAGE_GUIDE.md`

#### ...voir des exemples de code
→ Consultez `config_usage_examples.py`

#### ...comprendre les améliorations
→ Lancez `demo_config_benefits.py`

#### ...modifier les paramètres
→ Éditez `config/default_config.json` ou `config/colab_config.json`

#### ...créer un pipeline ML
→ Voir la section "Pipeline complet" dans `config_usage_examples.py`

#### ...utiliser Grad-CAM ou SHAP
→ Voir la section "Interprétabilité" dans `config_usage_examples.py`

#### ...faire une analyse exploratoire avancée (PCA, t-SNE, clustering)
→ Utilisez le template `notebooks/advanced_eda_template.ipynb`

## 🔗 Ressources complémentaires

### Documentation du projet:
- `README.md` (racine): Vue d'ensemble du projet
- `INSTALLATION.md`: Instructions d'installation
- `CHANGELOG_CONFIG_IMPROVEMENTS.md`: Détails des améliorations récentes

### Code source:
- `src/utils/config.py`: Implémentation de Config
- `config/default_config.json`: Configuration par défaut
- `config/colab_config.json`: Surcharges pour Colab
- `CELL_CONFIG_STANDALONE.py`: Cellule de configuration standalone

### Notebooks:
- `notebooks/`: Notebooks de démonstration
- `notebooks/colab/`: Notebooks spécifiques Colab

## ❓ FAQ

### Q: Dois-je encore utiliser `basic_usage.py`?
R: Non pour les nouveaux projets. Utilisez `CELL_CONFIG_STANDALONE.py` à la place.

### Q: Comment modifier les paramètres?
R: Éditez les fichiers JSON dans `config/`. Ne modifiez pas directement le code.

### Q: Quelle est la différence entre `default_config.json` et `colab_config.json`?
R: `default_config.json` est la base. `colab_config.json` override certains paramètres pour Google Colab.

### Q: Puis-je créer ma propre config?
R: Oui! Créez `config/my_config.json` et loadez-le avec `build_config(project_root, 'my')`.

### Q: Comment voir tous les paramètres disponibles?
R: Exécutez `print(config.to_dict())` après avoir chargé config.

### Q: Les exemples fonctionnent-ils sur Colab?
R: Oui! `CELL_CONFIG_STANDALONE.py` s'adapte automatiquement à Colab.

## 🤝 Contribuer

Pour ajouter de nouveaux exemples:
1. Créez un fichier `.py` descriptif
2. Ajoutez-le dans ce README
3. Incluez des commentaires clairs
4. Testez sur WSL ET Colab

## 📞 Support

- Issues: Créez une issue sur GitHub
- Questions: Consultez d'abord ce README et `CONFIG_USAGE_GUIDE.md`
- Documentation: Tous les fichiers sont commentés

---

**Dernière mise à jour**: Novembre 2025  
**Mainteneur**: Rafael (branche `rafael_cleaning`)
