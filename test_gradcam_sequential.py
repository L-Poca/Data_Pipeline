"""
Test de Grad-CAM avec un modèle Sequential simple.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from src.interpretability.gradcam import GradCAM

print("=" * 70)
print("TEST GRAD-CAM AVEC MODÈLE SEQUENTIAL")
print("=" * 70)

# Créer un modèle Sequential simple
print("\n1. Création du modèle Sequential...")

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
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
print(f"   Layers: {len(model.layers)}")

# Afficher l'architecture
print("\n2. Architecture du modèle:")
model.summary()

# Tester Grad-CAM
print("\n3. Test de Grad-CAM...")

try:
    # Initialiser Grad-CAM
    gradcam = GradCAM(model)
    print(f"✅ Grad-CAM initialisé")
    print(f"   Couche détectée: {gradcam.layer_name}")
    
    # Lister les couches disponibles
    conv_layers = gradcam.get_available_layers()
    print(f"\n   Couches Conv2D disponibles: {len(conv_layers)}")
    for i, layer_name in enumerate(conv_layers):
        print(f"      {i+1}. {layer_name}")
    
    # Créer une image de test
    print("\n4. Génération d'une image de test...")
    test_image = np.random.rand(224, 224, 3).astype('float32') * 255
    
    # Calculer la heatmap
    print("\n5. Calcul de la heatmap...")
    heatmap = gradcam.compute_heatmap(test_image, class_idx=0)
    
    print(f"✅ Heatmap calculée")
    print(f"   Shape: {heatmap.shape}")
    print(f"   Range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    # Test avec toutes les couches
    print("\n6. Test avec chaque couche Conv2D...")
    for layer_name in conv_layers:
        gradcam_layer = GradCAM(model, layer_name=layer_name)
        heatmap_layer = gradcam_layer.compute_heatmap(test_image, class_idx=0)
        print(f"   ✅ {layer_name:30s} → Heatmap {heatmap_layer.shape}")
    
    print("\n" + "=" * 70)
    print("🎉 TOUS LES TESTS SONT PASSÉS!")
    print("=" * 70)
    print("\n✅ Grad-CAM fonctionne correctement avec les modèles Sequential")
    
except Exception as e:
    print("\n" + "=" * 70)
    print("❌ ERREUR DÉTECTÉE")
    print("=" * 70)
    print(f"\nType: {type(e).__name__}")
    print(f"Message: {e}")
    
    import traceback
    print("\nTraceback complet:")
    traceback.print_exc()
