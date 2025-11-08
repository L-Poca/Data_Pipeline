# Exemple de cellule pour comprehensive_ml_pipeline_v2.ipynb
# À ajouter après l'entraînement des modèles

"""
=============================================================================
INTERPRÉTABILITÉ - GRAD-CAM, LIME & SHAP
=============================================================================

Cette section analyse l'interprétabilité des meilleurs modèles avec 3 techniques :
1. Grad-CAM : Cartes de chaleur des régions importantes
2. LIME : Explications avec super-pixels
3. SHAP : Valeurs de Shapley (optionnel, lent)
"""

print("=" * 70)
print("ANALYSE D'INTERPRÉTABILITÉ")
print("=" * 70)

from src.notebooks import run_full_interpretability_analysis, get_preprocessing_function

# Créer le dossier de résultats
interpretability_dir = config.results_dir / 'interpretability_analysis'
interpretability_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. INTERPRÉTABILITÉ DU MEILLEUR MODÈLE CNN
# =============================================================================

if cnn_results:
    print("\n" + "=" * 70)
    print("INTERPRÉTABILITÉ - MEILLEUR CNN")
    print("=" * 70)
    
    best_cnn_name = max(cnn_results.keys(), key=lambda k: cnn_results[k]['test_acc'])
    best_cnn_model = cnn_results[best_cnn_name]['model']
    
    print(f"Modèle: {best_cnn_name}")
    print(f"Accuracy: {cnn_results[best_cnn_name]['test_acc']:.4f}")
    
    # Prédictions
    y_pred_cnn = np.argmax(best_cnn_model.predict(X_test_cnn, verbose=0), axis=1)
    
    # Analyse complète (Grad-CAM + LIME)
    run_full_interpretability_analysis(
        model=best_cnn_model,
        x_data=X_test_cnn,
        y_true=y_test,
        y_pred=y_pred_cnn,
        class_names=config.classes,
        background_data=X_train[:50],  # Pour SHAP (optionnel)
        n_samples=2,  # 2 échantillons par classe
        strategy="one_per_class",  # 1 correct par classe
        save_dir=interpretability_dir / 'best_cnn',
        use_gradcam=True,
        use_lime=True,
        use_shap=False,  # SHAP désactivé (trop lent)
    )

# =============================================================================
# 2. INTERPRÉTABILITÉ DU MEILLEUR MODÈLE TRANSFER LEARNING
# =============================================================================

if transfer_results:
    print("\n" + "=" * 70)
    print("INTERPRÉTABILITÉ - MEILLEUR TRANSFER LEARNING")
    print("=" * 70)
    
    best_tl_name = max(transfer_results.keys(), key=lambda k: transfer_results[k]['test_acc'])
    best_tl_model = transfer_results[best_tl_name]['model']
    
    print(f"Modèle: {best_tl_name}")
    print(f"Accuracy: {transfer_results[best_tl_name]['test_acc']:.4f}")
    
    # Prédictions
    y_pred_tl = np.argmax(best_tl_model.predict(X_test_rgb, verbose=0), axis=1)
    
    # Fonction de preprocessing spécifique au modèle
    preprocess_fn = get_preprocessing_function(best_tl_name)
    
    # Analyse complète
    run_full_interpretability_analysis(
        model=best_tl_model,
        x_data=X_test_rgb,
        y_true=y_test,
        y_pred=y_pred_tl,
        class_names=config.classes,
        background_data=X_train_rgb[:50],
        n_samples=2,
        strategy="one_per_class",
        save_dir=interpretability_dir / 'best_transfer_learning',
        use_gradcam=True,
        use_lime=True,
        use_shap=False,
        preprocess_fn=preprocess_fn  # Important pour Transfer Learning
    )

# =============================================================================
# 3. ANALYSE D'ERREURS (Optionnel)
# =============================================================================

print("\n" + "=" * 70)
print("ANALYSE D'ERREURS")
print("=" * 70)

# Analyser les erreurs du meilleur modèle
if cnn_results:
    print("\n--- Erreurs du meilleur CNN ---")
    
    # Prédictions
    y_pred_cnn = np.argmax(best_cnn_model.predict(X_test_cnn, verbose=0), axis=1)
    
    # Trouver les erreurs
    errors_mask = y_pred_cnn != y_test
    n_errors = errors_mask.sum()
    
    print(f"Nombre d'erreurs: {n_errors}/{len(y_test)} ({n_errors/len(y_test)*100:.1f}%)")
    
    if n_errors > 0:
        # Analyser quelques erreurs
        run_full_interpretability_analysis(
            model=best_cnn_model,
            x_data=X_test_cnn,
            y_true=y_test,
            y_pred=y_pred_cnn,
            class_names=config.classes,
            n_samples=2,
            strategy="incorrect",  # Seulement les erreurs
            save_dir=interpretability_dir / 'error_analysis',
            use_gradcam=True,
            use_lime=False,  # Désactiver LIME pour aller plus vite
            use_shap=False,
        )

# =============================================================================
# 4. RÉSUMÉ DES RÉSULTATS D'INTERPRÉTABILITÉ
# =============================================================================

print("\n" + "=" * 70)
print("RÉSUMÉ - INTERPRÉTABILITÉ")
print("=" * 70)

print(f"\n📁 Résultats sauvegardés dans: {interpretability_dir}")
print("\nTechniques utilisées:")
print("  ✅ Grad-CAM: Cartes de chaleur des régions importantes")
print("  ✅ LIME: Explications avec super-pixels")
print("  ⏭️  SHAP: Désactivé (trop lent)")

print("\n📊 Analyses réalisées:")
if cnn_results:
    print(f"  • Meilleur CNN: {best_cnn_name}")
if transfer_results:
    print(f"  • Meilleur Transfer Learning: {best_tl_name}")
print("  • Analyse d'erreurs")

print("\n🔍 Insights clés à vérifier:")
print("  1. Le modèle se concentre-t-il sur les poumons ?")
print("  2. Les zones d'attention correspondent-elles aux symptômes ?")
print("  3. Y a-t-il des biais (artifacts, marqueurs) ?")
print("  4. Les erreurs sont-elles expliquables ?")

print("\n💡 Pour plus de détails:")
print("  • Consultez examples/README_INTERPRETABILITY.md")
print("  • Exécutez examples/interpretability_example.py")

print("\n✅ Analyse d'interprétabilité terminée!")
