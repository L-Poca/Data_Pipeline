"""
Notebook utilities module.

Provides reusable functions for Jupyter notebooks:
- Data loading and preprocessing
- Custom CNN and Transfer Learning model building
- Training and evaluation
- Visualization and interpretability
"""

from .notebook_utils import (
    # Data loading & preprocessing
    load_dataset,
    create_preprocessing_pipeline,
    prepare_train_val_test_split,
    compute_class_weights,
    create_data_generators,
    
    # Model building - Custom CNN
    build_custom_cnn,
    compile_model,
    create_callbacks,
    
    # Model building - Transfer Learning
    build_transfer_learning_model,
    create_transfer_learning_generators,
    unfreeze_top_layers,
    
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
    # Data
    'load_dataset',
    'create_preprocessing_pipeline',
    'prepare_train_val_test_split',
    'compute_class_weights',
    'create_data_generators',
    
    # Custom CNN
    'build_custom_cnn',
    'compile_model',
    'create_callbacks',
    
    # Transfer Learning
    'build_transfer_learning_model',
    'create_transfer_learning_generators',
    'unfreeze_top_layers',
    
    # Training
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
