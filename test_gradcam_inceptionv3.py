"""
Test de Grad-CAM avec InceptionV3 (pré-entraîné).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import InceptionV3
from src.interpretability.gradcam import GradCAM

print("=" * 70)
print("TEST GRAD-CAM AVEC INCEPTIONV3")
print("=" * 70)

# Créer un modèle avec InceptionV3
print("\n1. Chargement d'InceptionV3 pré-entraîné...")

base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Créer un modèle Sequential avec InceptionV3 + couches finales
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Modèle créé")
print(f"   Type: {type(model)}")
print(f"   Base model: {type(base_model)}")
print(f"   Total layers: {len(model.layers)}")
print(f"   InceptionV3 layers: {len(base_model.layers)}")

# Afficher un résumé simplifié
print("\n2. Architecture du modèle (simplifié):")
model.build((None, 224, 224, 3))  # Construire le modèle pour avoir output_shape
for i, layer in enumerate(model.layers):
    output_info = str(layer.output_shape) if hasattr(layer, 'output_shape') else "N/A"
    print(f"   {i}. {layer.name:40s} {str(layer.__class__.__name__):20s} → {output_info}")

# Tester Grad-CAM
print("\n3. Test de Grad-CAM (auto-détection)...")

try:
    # Initialiser Grad-CAM (auto-détection de la dernière Conv2D)
    gradcam = GradCAM(model)
    print(f"✅ Grad-CAM initialisé")
    print(f"   Couche détectée: {gradcam.layer_name}")
    
    # Lister les couches Conv2D disponibles
    print("\n4. Liste des couches Conv2D disponibles...")
    conv_layers = gradcam.get_available_layers()
    print(f"   Total: {len(conv_layers)} couches Conv2D")
    
    # Afficher les 10 premières et 10 dernières
    print("\n   Premières couches:")
    for i, layer_name in enumerate(conv_layers[:10]):
        print(f"      {i+1}. {layer_name}")
    
    if len(conv_layers) > 20:
        print(f"\n   ... ({len(conv_layers) - 20} couches intermédiaires) ...")
    
    print("\n   Dernières couches:")
    for i, layer_name in enumerate(conv_layers[-10:]):
        print(f"      {len(conv_layers) - 10 + i + 1}. {layer_name}")
    
    # Créer une image de test
    print("\n5. Génération d'une image de test...")
    test_image = np.random.rand(224, 224, 3).astype('float32') * 255
    
    # Calculer la heatmap
    print("\n6. Calcul de la heatmap...")
    heatmap = gradcam.compute_heatmap(test_image, class_idx=0)
    
    print(f"✅ Heatmap calculée")
    print(f"   Shape: {heatmap.shape}")
    print(f"   Range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    print(f"   Mean: {heatmap.mean():.3f}")
    print(f"   Std: {heatmap.std():.3f}")
    
    # Test avec différentes couches spécifiques
    print("\n7. Test avec des couches spécifiques...")
    
    # Tester quelques couches Conv2D importantes d'InceptionV3
    test_layers = []
    
    # Chercher des couches mixed (inception blocks)
    for layer_name in conv_layers:
        if 'mixed' in layer_name:
            test_layers.append(layer_name)
    
    # Prendre un échantillon
    if test_layers:
        sample_layers = test_layers[::len(test_layers)//5 + 1][:5]  # Max 5 couches
        
        for layer_name in sample_layers:
            gradcam_layer = GradCAM(model, layer_name=layer_name)
            heatmap_layer = gradcam_layer.compute_heatmap(test_image, class_idx=0)
            print(f"   ✅ {layer_name:50s} → Heatmap {heatmap_layer.shape}")
    
    print("\n" + "=" * 70)
    print("🎉 TOUS LES TESTS SONT PASSÉS!")
    print("=" * 70)
    print("\n✅ Grad-CAM fonctionne correctement avec InceptionV3")
    print("   - Modèle pré-entraîné avec sous-modèle")
    print("   - Détection automatique de la dernière Conv2D")
    print("   - Navigation dans les couches imbriquées")
    print("   - Calcul de heatmaps pour différentes couches")
    
except Exception as e:
    print("\n" + "=" * 70)
    print("❌ ERREUR DÉTECTÉE")
    print("=" * 70)
    print(f"\nType: {type(e).__name__}")
    print(f"Message: {e}")
    
    import traceback
    print("\nTraceback complet:")
    traceback.print_exc()
