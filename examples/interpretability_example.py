"""
Exemple d'utilisation des fonctions d'interprétabilité (Grad-CAM, LIME, SHAP)

Ce script montre comment utiliser les fonctions d'interprétabilité disponibles
dans src.notebooks.interpretability_utils pour analyser les prédictions des modèles.

Author: Data Pipeline Team
Date: November 2025
"""

from pathlib import Path
import numpy as np
from keras.models import load_model

from src.notebooks import (
    # Data loading
    load_dataset,
    prepare_train_val_test_split,
    # Interpretability
    setup_interpretability,
    run_gradcam_analysis,
    setup_lime_explainer,
    run_lime_analysis,
    setup_shap_explainer,
    run_shap_analysis,
    select_sample_images,
    run_full_interpretability_analysis,
    get_preprocessing_function,
)


def main():
    """Exemple d'analyse d'interprétabilité complète."""
    
    # =============================================================================
    # 1. CHARGEMENT DES DONNÉES
    # =============================================================================
    
    print("=" * 70)
    print("CHARGEMENT DES DONNÉES")
    print("=" * 70)
    
    # Charger le dataset
    data_dir = Path("data/raw/COVID-19_Radiography_Dataset")
    images, labels, class_names = load_dataset(
        data_dir=data_dir,
        img_size=(256, 256),
        n_images_per_class=100  # Limiter pour l'exemple
    )
    
    # Split train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_split(
        images, labels,
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    print(f"\nDataset chargé:")
    print(f"  • Train: {len(X_train)} images")
    print(f"  • Val: {len(X_val)} images")
    print(f"  • Test: {len(X_test)} images")
    print(f"  • Classes: {class_names}")
    
    # =============================================================================
    # 2. CHARGEMENT DU MODÈLE
    # =============================================================================
    
    print("\n" + "=" * 70)
    print("CHARGEMENT DU MODÈLE")
    print("=" * 70)
    
    # Charger un modèle pré-entraîné
    model_path = Path("models/inceptionv3_best.keras")
    
    if not model_path.exists():
        print(f"⚠️  Modèle non trouvé: {model_path}")
        print("   Veuillez d'abord entraîner un modèle.")
        return
    
    model = load_model(model_path)
    print(f"✅ Modèle chargé: {model_path}")
    
    # Faire des prédictions
    print("\n   Prédictions sur le test set...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    accuracy = np.mean(y_pred == y_test)
    print(f"   Accuracy: {accuracy:.2%}")
    
    # =============================================================================
    # 3. MÉTHODE 1 : ANALYSE COMPLÈTE (RECOMMANDÉ)
    # =============================================================================
    
    print("\n" + "=" * 70)
    print("MÉTHODE 1: ANALYSE D'INTERPRÉTABILITÉ COMPLÈTE")
    print("=" * 70)
    
    results_dir = Path("results/interpretability_example")
    
    # Fonction de preprocessing pour InceptionV3
    preprocess_fn = get_preprocessing_function('InceptionV3')
    
    # Convertir images en RGB pour InceptionV3
    X_test_rgb = np.repeat(X_test[..., np.newaxis], 3, axis=-1)
    
    # Lancer l'analyse complète (Grad-CAM + LIME, SHAP optionnel)
    run_full_interpretability_analysis(
        model=model,
        x_data=X_test_rgb,
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        background_data=X_train[:50],  # Pour SHAP
        n_samples=1,  # 1 échantillon par classe
        strategy="one_per_class",
        save_dir=results_dir,
        use_gradcam=True,
        use_lime=True,
        use_shap=False,  # SHAP est lent, désactivé par défaut
        preprocess_fn=preprocess_fn
    )
    
    # =============================================================================
    # 4. MÉTHODE 2 : ANALYSE SÉPARÉE PAR TECHNIQUE
    # =============================================================================
    
    print("\n" + "=" * 70)
    print("MÉTHODE 2: ANALYSES SÉPARÉES")
    print("=" * 70)
    
    # Sélectionner des échantillons
    indices, descriptions = select_sample_images(
        X_test_rgb, y_test, y_pred, class_names,
        n_samples=2,
        strategy="correct"
    )
    
    # --- Grad-CAM uniquement ---
    print("\n--- Grad-CAM ---")
    gradcam = setup_interpretability(model, verbose=True)
    run_gradcam_analysis(
        gradcam, X_test_rgb, indices, descriptions, class_names,
        y_pred_probs=y_pred_probs,
        save_dir=results_dir / "gradcam_only",
        preprocess_fn=preprocess_fn
    )
    
    # --- LIME uniquement ---
    print("\n--- LIME ---")
    lime_explainer = setup_lime_explainer(
        model,
        segmentation_method='quickshift',
        num_samples=500,  # Moins d'échantillons pour plus de rapidité
        verbose=True
    )
    
    if lime_explainer:
        run_lime_analysis(
            lime_explainer, X_test_rgb, indices, descriptions, class_names,
            y_pred=y_pred,
            num_features=5,
            save_dir=results_dir / "lime_only"
        )
    
    # --- SHAP uniquement (optionnel, lent) ---
    print("\n--- SHAP ---")
    use_shap = False  # Mettre à True pour activer SHAP
    
    if use_shap:
        shap_explainer = setup_shap_explainer(
            model,
            background_data=X_train[:50],
            max_background_samples=50,
            verbose=True
        )
        
        if shap_explainer:
            run_shap_analysis(
                shap_explainer, X_test_rgb, indices[:2], descriptions[:2], class_names,
                y_pred=y_pred,
                save_dir=results_dir / "shap_only"
            )
    else:
        print("⚠️  SHAP désactivé (lent). Mettre use_shap=True pour activer.")
    
    # =============================================================================
    # 5. ANALYSE AVANCÉE : COMPARAISONS
    # =============================================================================
    
    print("\n" + "=" * 70)
    print("ANALYSES AVANCÉES")
    print("=" * 70)
    
    # Comparer les méthodes de segmentation LIME
    if lime_explainer:
        from src.notebooks.interpretability_utils import compare_lime_segmentation
        
        print("\n--- Comparaison des segmentations LIME ---")
        compare_lime_segmentation(
            lime_explainer,
            X_test_rgb[indices[0]],
            save_dir=results_dir / "lime_comparison"
        )
    
    # Comparer les classes avec SHAP
    if use_shap and shap_explainer:
        from src.notebooks.interpretability_utils import compare_shap_classes
        
        print("\n--- Comparaison SHAP inter-classes ---")
        compare_shap_classes(
            shap_explainer,
            X_test_rgb[indices[0]],
            class_names,
            save_dir=results_dir / "shap_comparison"
        )
    
    # =============================================================================
    # CONCLUSION
    # =============================================================================
    
    print("\n" + "=" * 70)
    print("✅ ANALYSE D'INTERPRÉTABILITÉ TERMINÉE")
    print("=" * 70)
    print(f"\n📁 Résultats sauvegardés dans: {results_dir}")
    print("\nRésumé des techniques:")
    print("  • Grad-CAM: Visualisation des zones importantes pour les CNN")
    print("  • LIME: Explications locales avec super-pixels")
    print("  • SHAP: Valeurs de Shapley pour importance des pixels")
    print("\nPour plus d'informations, consultez la documentation dans:")
    print("  • src/interpretability/gradcam.py")
    print("  • src/interpretability/lime_explainer.py")
    print("  • src/interpretability/shap_explainer.py")


if __name__ == "__main__":
    main()
