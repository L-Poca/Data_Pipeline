"""
Test minimal pour vérifier que les fonctions d'interprétabilité fonctionnent.

Ce script teste les imports et la configuration de base sans nécessiter de données.
"""

import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TEST MINIMAL - INTERPRETABILITY UTILS")
print("=" * 70)

# =============================================================================
# TEST 1: IMPORTS
# =============================================================================

print("\n[1/4] Test des imports...")

try:
    from src.notebooks import (
        # Grad-CAM
        setup_interpretability,
        run_gradcam_analysis,
        get_preprocessing_function,
        # LIME
        setup_lime_explainer,
        run_lime_analysis,
        compare_lime_segmentation,
        # SHAP
        setup_shap_explainer,
        run_shap_analysis,
        compare_shap_classes,
        # Complet
        run_full_interpretability_analysis,
        select_sample_images,
    )
    print("   ✅ Tous les imports réussis")
except Exception as e:
    print(f"   ❌ Erreur d'import: {e}")
    exit(1)

# =============================================================================
# TEST 2: PREPROCESSING FUNCTIONS
# =============================================================================

print("\n[2/4] Test des fonctions de preprocessing...")

try:
    models_to_test = ['InceptionV3', 'ResNet50', 'VGG16', 'EfficientNetB0']
    for model_name in models_to_test:
        preprocess_fn = get_preprocessing_function(model_name)
        assert preprocess_fn is not None
    print(f"   ✅ {len(models_to_test)} fonctions de preprocessing disponibles")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    exit(1)

# =============================================================================
# TEST 3: VÉRIFICATION LIME
# =============================================================================

print("\n[3/4] Test de disponibilité LIME...")

try:
    import lime
    try:
        version = lime.__version__
    except AttributeError:
        version = "installé"
    print(f"   ✅ LIME {version}")
except ImportError:
    print("   ⚠️  LIME non installé (pip install lime)")
    print("      Les fonctions LIME sont disponibles mais nécessitent l'installation")

# =============================================================================
# TEST 4: VÉRIFICATION SHAP
# =============================================================================

print("\n[4/4] Test de disponibilité SHAP...")

try:
    import shap
    try:
        version = shap.__version__
    except AttributeError:
        version = "installé"
    print(f"   ✅ SHAP {version}")
except ImportError:
    print("   ⚠️  SHAP non installé (pip install shap)")
    print("      Les fonctions SHAP sont disponibles mais nécessitent l'installation")

# =============================================================================
# RÉSUMÉ
# =============================================================================

print("\n" + "=" * 70)
print("RÉSUMÉ DES TESTS")
print("=" * 70)

print("\n✅ Fonctions disponibles:")
print("   • setup_interpretability (Grad-CAM)")
print("   • run_gradcam_analysis")
print("   • get_preprocessing_function")
print("   • setup_lime_explainer")
print("   • run_lime_analysis")
print("   • compare_lime_segmentation")
print("   • setup_shap_explainer")
print("   • run_shap_analysis")
print("   • compare_shap_classes")
print("   • run_full_interpretability_analysis")
print("   • select_sample_images")

print("\n📦 Dépendances:")
try:
    import lime
    print("   ✅ LIME: Installé")
except ImportError:
    print("   ⏭️  LIME: Non installé (optionnel)")

try:
    import shap
    print("   ✅ SHAP: Installé")
except ImportError:
    print("   ⏭️  SHAP: Non installé (optionnel)")

print("\n🎯 Prochaines étapes:")
print("   1. Installer les dépendances optionnelles:")
print("      pip install lime shap")
print("   2. Consulter la documentation:")
print("      cat examples/README_INTERPRETABILITY.md")
print("   3. Tester avec un exemple complet:")
print("      python examples/interpretability_example.py")
print("   4. Intégrer dans votre notebook:")
print("      Copier CELL_INTERPRETABILITY.py dans votre notebook")

print("\n" + "=" * 70)
print("✅ TOUS LES TESTS RÉUSSIS")
print("=" * 70)
print("\nLe module interpretability_utils est prêt à l'emploi ! 🎉")
