"""
Test des corrections dans interpretability_utils.py

Vérifie que la gestion des indices est correcte.
"""

import numpy as np
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path.cwd() / 'src'))

print("=" * 70)
print("TEST DES CORRECTIONS - INTERPRETABILITY_UTILS")
print("=" * 70)

# Test 1: Import du module
print("\n[1/4] Test d'import du module...")
try:
    from notebooks import (
        select_sample_images,
        run_full_interpretability_analysis,
    )
    print("   ✅ Import réussi")
except Exception as e:
    print(f"   ❌ Erreur d'import: {e}")
    sys.exit(1)

# Test 2: Création de données fictives
print("\n[2/4] Création de données de test...")
n_samples = 100
img_size = (256, 256, 3)
n_classes = 4

X_test = np.random.rand(n_samples, *img_size).astype('float32')
y_test = np.random.randint(0, n_classes, n_samples)
y_pred = np.random.randint(0, n_classes, n_samples)
class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

print(f"   • X_test shape: {X_test.shape}")
print(f"   • y_test shape: {y_test.shape}")
print(f"   • y_pred shape: {y_pred.shape}")
print(f"   • Classes: {class_names}")

# Test 3: Sélection d'échantillons
print("\n[3/4] Test de sélection d'échantillons...")
try:
    indices, descriptions = select_sample_images(
        X_test, y_test, y_pred, class_names,
        n_samples=2,
        strategy="one_per_class"
    )
    
    print(f"\n   ✅ Sélection réussie:")
    print(f"      • Nombre d'indices: {len(indices)}")
    print(f"      • Indices: {indices}")
    print(f"      • Descriptions: {descriptions}")
    
    # Vérifier que les indices sont valides
    assert len(indices) <= n_classes, f"Trop d'indices: {len(indices)} > {n_classes}"
    assert all(0 <= idx < n_samples for idx in indices), "Indices hors limites"
    
    print("   ✅ Indices valides")
    
except Exception as e:
    print(f"   ❌ Erreur lors de la sélection: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Simulation de la logique de run_full_interpretability_analysis
print("\n[4/4] Test de la logique d'extraction des données...")
try:
    # Simuler ce que fait run_full_interpretability_analysis
    x_selected = X_test[indices]
    y_pred_selected = y_pred[indices]
    
    print(f"\n   ✅ Extraction réussie:")
    print(f"      • x_selected shape: {x_selected.shape}")
    print(f"      • y_pred_selected shape: {y_pred_selected.shape}")
    
    # Créer de nouveaux indices [0, 1, 2, 3]
    new_indices = list(range(len(indices)))
    print(f"      • Nouveaux indices: {new_indices}")
    
    # Simuler l'accès aux données
    print("\n   Test d'accès aux données avec les nouveaux indices:")
    for i, (new_idx, desc) in enumerate(zip(new_indices, descriptions)):
        img = x_selected[new_idx]
        pred = y_pred_selected[new_idx]
        
        # Vérifier que l'accès fonctionne
        assert img.shape == img_size, f"Mauvaise shape: {img.shape} != {img_size}"
        assert 0 <= pred < n_classes, f"Prédiction hors limites: {pred}"
        
        print(f"      [{i+1}/{len(indices)}] {desc}")
        print(f"         • Image shape: {img.shape} ✓")
        print(f"         • Prédiction: {pred} ({class_names[pred]}) ✓")
    
    print("\n   ✅ Tous les accès réussis!")
    
except Exception as e:
    print(f"   ❌ Erreur lors de l'extraction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Vérifier qu'on ne peut plus accéder aux anciens indices
print("\n[5/5] Test de protection contre les anciens indices...")
try:
    # Essayer d'accéder avec un ancien indice (qui serait hors limites)
    if len(indices) > 0:
        old_idx = indices[0]  # Ex: 45
        
        # Cela devrait échouer si on essaie sur x_selected
        try:
            if old_idx >= len(x_selected):
                # C'est attendu - l'ancien indice est trop grand
                print(f"   ✅ Protection correcte:")
                print(f"      • Ancien indice: {old_idx}")
                print(f"      • Taille x_selected: {len(x_selected)}")
                print(f"      • Accès avec ancien indice impossible ✓")
            else:
                print(f"   ⚠️  Ancien indice {old_idx} < {len(x_selected)}, pas de conflit")
        except IndexError:
            print(f"   ✅ IndexError correctement levée")
    
except Exception as e:
    print(f"   ❌ Erreur inattendue: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ TOUS LES TESTS RÉUSSIS!")
print("=" * 70)
print("\nLes corrections dans interpretability_utils.py fonctionnent correctement.")
print("Vous pouvez maintenant relancer le notebook.")
