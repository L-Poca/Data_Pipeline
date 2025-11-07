# 🧠 Custom CNN with Interpretability - Documentation

## Overview

This notebook (`custom_cnn_with_interpretability.ipynb`) provides a complete pipeline for training a custom deep CNN for COVID-19 radiography classification with comprehensive interpretability analysis using Grad-CAM, LIME, and SHAP.

**Note**: This is a template notebook with unexecuted cells (`execution_count: null`). All cells will populate with execution numbers and outputs when you run them in Jupyter/Colab.

## Features

### ✅ Complete Data Pipeline
- Automatic environment detection (Colab/WSL/Local)
- Data loading using the project's pipeline transformers
- Preprocessing with `ImageLoader` and `ImageResizer`
- Train/Val/Test split (70/15/15) with stratification
- Class weight computation for balanced learning

### ✅ Robust Custom CNN Architecture
- **5 convolutional blocks** with progressive filter increases (32→64→128→256→512)
- **Batch Normalization** for training stability
- **Dropout layers** for regularization (0.25→0.5)
- **L2 regularization** on dense layers
- **~15M trainable parameters** optimized for Colab Pro GPU

### ✅ Advanced Training
- Adam optimizer with adaptive learning rate
- Comprehensive metrics: accuracy, AUC, precision, recall
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Data augmentation**: rotation, shift, zoom, horizontal flip
- **Class weighting** for handling class imbalance

### ✅ Complete Interpretability Suite
- **Grad-CAM**: Fast attention heatmaps showing where the model looks
- **LIME**: Super-pixel based explanations for local interpretability
- **SHAP**: Shapley values for rigorous pixel-level contributions
- Side-by-side comparison of all three methods

## How to Use

### On Google Colab (Recommended)

1. **Open the notebook in Colab**:
   - Click the "Open in Colab" badge at the top of the notebook
   - Or manually upload to Colab

2. **Enable GPU**:
   ```
   Runtime → Change runtime type → Hardware accelerator: GPU (T4 or better)
   ```

3. **Run all cells**:
   - The first cell automatically handles environment setup
   - Clones the repository
   - Installs dependencies
   - Mounts Google Drive for dataset access
   - Extracts the COVID-19 dataset

4. **Monitor training**:
   - Training takes ~30-60 minutes on Colab Pro with GPU
   - Watch the training curves in real-time
   - Best model is automatically saved

5. **Explore interpretability**:
   - Grad-CAM generates attention heatmaps instantly
   - LIME takes 2-3 minutes for explanations
   - SHAP takes 3-5 minutes for Shapley values

### On Local Machine / WSL

1. **Prerequisites**:
   ```bash
   # Ensure you have the repository cloned
   git clone https://github.com/L-Poca/Data_Pipeline.git
   cd Data_Pipeline
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Dataset**:
   - Ensure the COVID-19 Radiography dataset is in `data/raw/COVID-19_Radiography_Dataset/`
   - Dataset structure:
     ```
     COVID-19_Radiography_Dataset/
     ├── COVID/images/*.png
     ├── Normal/images/*.png
     ├── Lung_Opacity/images/*.png
     └── Viral Pneumonia/images/*.png
     ```

3. **Run the notebook**:
   ```bash
   jupyter notebook notebooks/colab/custom_cnn_with_interpretability.ipynb
   ```

## Notebook Structure

### Section 1: Initialization (Cells 1-7)
- Environment detection and setup
- GPU verification
- Import necessary libraries

### Section 2: Data Loading (Cells 8-12)
- Load images using pipeline
- Preprocessing and normalization
- Train/Val/Test split
- Class weight computation

### Section 3: Data Augmentation (Cell 13)
- Configure ImageDataGenerator
- Create training and validation generators

### Section 4: CNN Architecture (Cells 14-18)
- Define custom CNN architecture
- Build model
- Display model summary and parameters

### Section 5: Training (Cells 19-24)
- Compile model with optimizer and metrics
- Configure callbacks
- Train model with class weights
- Visualize training curves

### Section 6-7: Evaluation (Cells 25-29)
- Evaluate on test set
- Generate confusion matrix
- Display classification report

### Section 8: Interpretability (Cells 30-50)
- Import interpretability modules
- Select sample images
- **Grad-CAM**: Generate attention heatmaps
- **LIME**: Create super-pixel explanations
- **SHAP**: Compute Shapley values
- Compare all three methods side-by-side

### Section 9: Summary (Cells 51-57)
- Key insights and findings
- Comparison table of interpretability methods
- Generated files and next steps

## Output Files

All results are saved in the `results/` directory:

```
results/
├── custom_cnn_models/
│   └── best_model.keras              # Best model checkpoint
├── custom_cnn_plots/
│   ├── training_curves.png          # Loss, accuracy, AUC curves
│   └── confusion_matrix.png         # Confusion matrix heatmap
└── interpretability_custom_cnn/
    ├── gradcam_grid.png             # Grad-CAM visualizations
    ├── lime_explanations.png        # LIME explanations
    ├── shap_explanations.png        # SHAP visualizations
    └── comparison_all_methods.png   # Side-by-side comparison
```

## Expected Results

### Training Performance
- **Training time**: 
  - Colab Free (T4 GPU): 60-90 minutes
  - Colab Pro (V100/A100 GPU): 20-40 minutes
  - Local GPU (varies): 30-60 minutes
- **Expected accuracy**: 85-95% on test set
- **Convergence**: Usually within 20-30 epochs with early stopping

### Model Architecture
- **Input**: 224×224×3 RGB images
- **Output**: 4 classes (COVID, Normal, Lung_Opacity, Viral Pneumonia)
- **Parameters**: ~15M trainable parameters
- **Model size**: ~60 MB

### Interpretability Timing
- **Grad-CAM**: <1 second per image
- **LIME**: ~30-60 seconds per image
- **SHAP**: ~60-120 seconds per image

## Customization

### Adjust Model Complexity
```python
# In cell 14 (build_custom_cnn function)
# Modify filter sizes or add/remove blocks
# Example: Change 512 to 256 for lighter model
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
```

### Change Training Parameters
```python
# In cell 8: Number of images per class
N_IMAGES_PER_CLASS = 1000  # or None for all images

# In cell 19: Learning rate
optimizer = Adam(learning_rate=0.0001)  # Lower for fine-tuning

# In cell 24: Number of epochs
EPOCHS = 100  # Increase for more training
```

### Modify Data Augmentation
```python
# In cell 13: Adjust augmentation parameters
train_datagen = ImageDataGenerator(
    rotation_range=20,        # Increase rotation
    width_shift_range=0.2,    # Increase shift
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,       # Add vertical flip
    zoom_range=0.15,          # Increase zoom
    fill_mode='nearest'
)
```

## Troubleshooting

### Out of Memory (OOM)
- **Reduce batch size**: In the data augmentation cell (cell 13), modify:
  ```python
  # Change the batch_size when creating generators
  train_generator = train_datagen.flow(
      X_train, y_train_cat,
      batch_size=16,  # Reduce from 32 to 16
      shuffle=True
  )
  ```
- **Reduce model size**: Use fewer filters in conv layers (cell 14)
- **Limit dataset**: Set `N_IMAGES_PER_CLASS = 500` in cell 8

### Slow Training
- **Enable GPU**: Ensure GPU is enabled in Colab
- **Reduce data augmentation**: Lower `num_samples` in LIME/SHAP
- **Use subset**: Train on fewer images for testing

### Import Errors
- **Reinstall packages**: 
  ```bash
  pip install -r requirements.txt --force-reinstall
  ```
- **Update package**: 
  ```bash
  pip install -e . --upgrade
  ```

## Comparison with Transfer Learning

This custom CNN can be compared with the transfer learning approach in `transfer_learning_colab_revamp.ipynb`:

| Aspect | Custom CNN | Transfer Learning |
|--------|-----------|------------------|
| **Training time** | 30-60 min | 20-40 min |
| **Parameters** | ~15M | 20-50M (depending on base model) |
| **Accuracy** | 85-95% | 90-98% |
| **Interpretability** | Full access to all layers | Limited by pre-trained architecture |
| **Flexibility** | Complete control | Constrained by base model |
| **Best for** | Learning, experimentation | Production, high accuracy |

## References

- **Grad-CAM**: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (2017)
- **LIME**: Ribeiro et al. "Why Should I Trust You?" (2016)
- **SHAP**: Lundberg & Lee "A Unified Approach to Interpreting Model Predictions" (2017)
- **Dataset**: COVID-19 Radiography Database (Kaggle)

## License

This notebook is part of the Data_Pipeline project and follows the same license.

## Contributing

For issues or improvements:
1. Open an issue on GitHub
2. Submit a pull request
3. Contact the maintainers

## Authors

Generated as part of the Data_Pipeline project for COVID-19 radiography classification with interpretability.

---

**Happy Training! 🚀**
