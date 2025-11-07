"""
Notebook utilities module.

Provides reusable functions for Jupyter notebooks including:
- Data loading and preprocessing
- Model building and training
- Evaluation and visualization
- Interpretability analysis
"""

from .notebook_utils import (
    # Data loading & preprocessing
    load_dataset,
    create_preprocessing_pipeline,
    prepare_train_val_test_split,
    compute_class_weights,
    create_data_generators,
    
    # Model building
    build_custom_cnn,
    compile_model,
    create_callbacks,
    
    # Training & evaluation
    train_model,
    evaluate_model,
    
    # Visualization
    plot_training_curves,
    plot_confusion_matrix,
    
    # Interpretability
    setup_interpretability,
    run_gradcam_analysis,
    select_sample_images,
)

__all__ = [
    # Data loading & preprocessing
    'load_dataset',
    'create_preprocessing_pipeline',
    'prepare_train_val_test_split',
    'compute_class_weights',
    'create_data_generators',
    
    # Model building
    'build_custom_cnn',
    'compile_model',
    'create_callbacks',
    
    # Training & evaluation
    'train_model',
    'evaluate_model',
    
    # Visualization
    'plot_training_curves',
    'plot_confusion_matrix',
    
    # Interpretability
    'setup_interpretability',
    'run_gradcam_analysis',
    'select_sample_images',
]
