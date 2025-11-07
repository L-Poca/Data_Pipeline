"""
╔═══════════════════════════════════════════════════════════════════════════╗
║  🔄 DÉMONSTRATION: AVANT vs APRÈS l'utilisation de config                 ║
╚═══════════════════════════════════════════════════════════════════════════╝

Ce script montre concrètement les améliorations apportées par l'usage
systématique de l'objet config.
"""

print("=" * 80)
print("❌ AVANT: Code avec variables dispersées")
print("=" * 80)

# Code typique AVANT l'amélioration
code_before = '''
# Configuration dispersée dans plusieurs variables
data_dir = Path("data/raw/COVID-19_Radiography_Dataset")
models_dir = Path("models")
results_dir = Path("results")

batch_size = 32
epochs = 50
learning_rate = 0.001
validation_split = 0.2

img_width = 256
img_height = 256
img_channels = 3

classes = ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]

# Dans le code d'entraînement
model.fit(
    X_train, y_train,
    batch_size=batch_size,  # Quelle variable utiliser?
    epochs=epochs,
    validation_split=validation_split
)

# Création de pipeline
pipeline = Pipeline([
    ('resizer', ImageResizer(target_size=(img_width, img_height))),
    ('normalizer', ImageNormalizer())
])

# Callbacks
early_stop = EarlyStopping(patience=10)  # Hardcodé!
reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)

# Visualisation
plt.figure(figsize=(12, 8))  # Valeurs hardcodées
plt.style.use('seaborn-v0_8')

# Grad-CAM
gradcam = GradCAM(model, alpha=0.4, colormap='jet')

# Problèmes:
# 1. Variables dispersées (difficiles à retrouver)
# 2. Valeurs hardcodées (difficiles à changer)
# 3. Duplication (batch_size défini plusieurs fois?)
# 4. Pas de source unique de vérité
# 5. Difficile de voir tous les paramètres d'un coup
'''

print(code_before)

print("\n" + "=" * 80)
print("✅ APRÈS: Code avec config centralisé")
print("=" * 80)

code_after = '''
# Une seule ligne pour charger TOUTE la configuration
from src.utils.config import build_config
config = build_config(project_root, ENV)

# Dans le code d'entraînement
model.fit(
    X_train, y_train,
    batch_size=config.batch_size,      # ✅ Source claire
    epochs=config.epochs,              # ✅ Facile à modifier dans JSON
    validation_split=config.validation_split
)

# Création de pipeline
pipeline = Pipeline([
    ('resizer', ImageResizer(target_size=config.img_size)),  # ✅ Tuple déjà formé
    ('normalizer', ImageNormalizer())
])

# Callbacks
early_stop = EarlyStopping(patience=config.early_stopping_patience)
reduce_lr = ReduceLROnPlateau(
    patience=config.reduce_lr_patience,
    factor=config.reduce_lr_factor,
    min_lr=config.min_lr
)

# Visualisation (automatique dans CELL_CONFIG_STANDALONE)
# plt.rcParams déjà configurés avec:
# - figure.figsize = config.figure_size
# - figure.dpi = config.dpi
# - style = config.plot_style

# Grad-CAM
gradcam = GradCAM(
    model,
    alpha=config.gradcam_alpha,
    colormap=config.gradcam_colormap
)

# Avantages:
# ✅ 1. Configuration centralisée dans config/
# ✅ 2. Tous les paramètres visibles d'un coup (config.to_dict())
# ✅ 3. Modification facile via JSON (pas besoin de toucher le code)
# ✅ 4. Adaptation automatique Colab/WSL
# ✅ 5. Source unique de vérité
# ✅ 6. Reproductibilité (config versionnée)
'''

print(code_after)

print("\n" + "=" * 80)
print("📊 COMPARAISON DES APPROCHES")
print("=" * 80)

comparison = """
┌─────────────────────────────┬─────────────────────┬─────────────────────┐
│ Critère                     │ ❌ AVANT            │ ✅ APRÈS            │
├─────────────────────────────┼─────────────────────┼─────────────────────┤
│ Nombre de variables         │ ~15-20 dispersées   │ 1 objet (config)    │
│ Modification paramètres     │ Changer le code     │ Éditer JSON         │
│ Visibilité paramètres       │ Dispersés partout   │ config.to_dict()    │
│ Adaptation Colab/WSL        │ Duplication code    │ Automatique         │
│ Reproductibilité            │ Difficile           │ JSON versionné      │
│ Maintenance                 │ Recherche manuelle  │ Un seul fichier     │
│ Documentation               │ Commentaires épars  │ Centralisée         │
│ Risque d'incohérence        │ Élevé               │ Faible              │
│ Facilité d'utilisation      │ Complexe            │ Simple              │
└─────────────────────────────┴─────────────────────┴─────────────────────┘
"""

print(comparison)

print("\n" + "=" * 80)
print("💡 EXEMPLES CONCRETS DE GAINS")
print("=" * 80)

examples = """
1️⃣ MODIFICATION D'HYPERPARAMÈTRES
   ❌ AVANT: Chercher batch_size=32 dans 10 notebooks différents
   ✅ APRÈS: Éditer config/default_config.json une seule fois

2️⃣ ADAPTATION COLAB
   ❌ AVANT: Dupliquer le code avec if ENV == "colab"
   ✅ APRÈS: config/colab_config.json override automatiquement

3️⃣ NOUVEAUX PARAMÈTRES
   ❌ AVANT: Ajouter une variable dans chaque notebook
   ✅ APRÈS: Ajouter dans config.py et default_config.json

4️⃣ DOCUMENTATION
   ❌ AVANT: Chercher dans le code les valeurs utilisées
   ✅ APRÈS: print(config.to_dict()) montre tout

5️⃣ REPRODUCTIBILITÉ
   ❌ AVANT: "Quels paramètres as-tu utilisés?" → Pas sûr...
   ✅ APRÈS: Partager config/default_config.json → Exactement les mêmes

6️⃣ REFACTORING
   ❌ AVANT: Remplacer 50 occurrences de img_width = 256
   ✅ APRÈS: Changer une ligne dans config.json

7️⃣ TESTS
   ❌ AVANT: Créer des variables de test dans chaque test
   ✅ APRÈS: Créer un config de test et l'utiliser partout

8️⃣ INTERPRÉTABILITÉ
   ❌ AVANT: gradcam_alpha=0.4 hardcodé → difficile de tester d'autres valeurs
   ✅ APRÈS: config.gradcam_alpha → changement centralisé
"""

print(examples)

print("\n" + "=" * 80)
print("🚀 GUIDE D'UTILISATION")
print("=" * 80)

guide = """
ÉTAPE 1: Copier CELL_CONFIG_STANDALONE.py en première cellule du notebook
         ↓
ÉTAPE 2: Exécuter la cellule → config est disponible globalement
         ↓
ÉTAPE 3: Utiliser config.xxx partout dans votre code
         ↓
ÉTAPE 4: Modifier les paramètres dans config/default_config.json si nécessaire

📝 RÈGLE D'OR:
   Toujours écrire:     config.batch_size
   Jamais écrire:       batch_size = 32
                        ou
                        batch_size = config.batch_size
"""

print(guide)

print("\n" + "=" * 80)
print("✅ CONCLUSION")
print("=" * 80)

conclusion = """
L'utilisation systématique de l'objet config apporte:
- 🎯 Centralisation: Un seul point de vérité
- 🔄 Flexibilité: Adaptation automatique des environnements
- 📊 Visibilité: Tous les paramètres accessibles facilement
- 🛠️ Maintenabilité: Changements simples et rapides
- 📝 Reproductibilité: Configuration versionnée et partageable
- 🚀 Productivité: Moins de temps perdu à chercher/modifier des paramètres

👉 Pour plus d'exemples: examples/config_usage_examples.py
👉 Documentation complète: examples/CONFIG_USAGE_GUIDE.md
"""

print(conclusion)
print("=" * 80)

if __name__ == "__main__":
    print("\n✨ Démonstration terminée!")
    print("💡 Adoptez l'objet config dans tous vos notebooks!")
