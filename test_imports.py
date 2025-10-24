#!/usr/bin/env python3
"""Script de test pour vérifier que les imports fonctionnent après la réorganisation."""

import sys
import os

# Ajouter le répertoire racine du projet au chemin Python
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("🔧 Test des imports après réorganisation...")
print(f"📁 Chemin du projet: {project_root}")

try:
    # Test des imports
    from src.features import (
        ImageLoader,
        ImageResizer,
        ImageNormalizer,
        ImageMasker,
        ImageFlattener,
        ImageBinarizer,
        ImageAugmenter,
        ImageRandomCropper,
        ImageHistogram,
        ImagePCA,
        ImageStandardScaler,
        VisualizeTransformer,
        SaveTransformer,
    )
    
    print("✅ Tous les transformateurs importés avec succès!")
    
    # Afficher la liste des transformateurs
    transformers = [
        ImageLoader, ImageResizer, ImageNormalizer, ImageMasker, ImageFlattener,
        ImageBinarizer, ImageAugmenter, ImageRandomCropper, ImageHistogram,
        ImagePCA, ImageStandardScaler, VisualizeTransformer, SaveTransformer
    ]
    
    print("\n📋 Transformateurs disponibles:")
    for transformer in transformers:
        print(f"  - {transformer.__name__}")
        
    print(f"\n🎉 Import réussi! {len(transformers)} transformateurs disponibles.")
    
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("🔍 Vérification de la structure des dossiers...")
    
    # Diagnostic
    features_path = os.path.join(project_root, 'src', 'features')
    pipelines_path = os.path.join(features_path, 'Pipelines')
    transformateurs_path = os.path.join(pipelines_path, 'Transformateurs')
    
    print(f"📁 src/features exists: {os.path.exists(features_path)}")
    print(f"📁 src/features/Pipelines exists: {os.path.exists(pipelines_path)}")
    print(f"📁 src/features/Pipelines/Transformateurs exists: {os.path.exists(transformateurs_path)}")
    
    if os.path.exists(transformateurs_path):
        print(f"📂 Contenu de Transformateurs: {os.listdir(transformateurs_path)}")