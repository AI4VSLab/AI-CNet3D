# 3D OCT Glaucoma Detection with Novel Cross-Attention Mechanisms, Multi-Task Consistency SSL Fine-tuning, and CARE Visualizations

This repository contains the implementation of **AICNet3D**, a novel 3D convolutional neural network architecture specifically designed for glaucoma detection from Optical Coherence Tomography (OCT) volumes. The model introduces innovative cross-attention mechanisms that leverage the anatomical structure of the optic nerve head (ONH) and macula, multi-task consistency SSL fine-tuning, and CARE visualizations to improve diagnostic accuracy and interpretability while remaining parameter-efficient.

## Key Novel Contributions

### 1. Novel Cross-Attention Mechanisms

The project introduces three novel cross-attention mechanisms that exploit the anatomical structure of the optic nerve head:

1. **HemiretinalCrossAttentionBlock** - Captures interactions between superior and inferior hemiretinas
2. **NeuronAxonCrossAttentionBlock** - Models relationships between neuronal and axonal regions  
3. **HemiretinalNeuronAxonCrossAttentionBlock** - Combines both hemiretinal and neuron-axon attention patterns

### 2. CARE (Channel Attention REpresentations) Visualization

**CARE** is a novel attention visualization technique that extracts and visualizes the learned attention patterns from the cross-attention blocks. Unlike traditional Grad-CAM which relies on gradients, CARE directly captures the attention weights from the anatomically-informed cross-attention mechanisms:

- **Direct Attention Extraction**: CARE extracts attention maps directly from the cross-attention layers without requiring gradients
- **Anatomical Relevance**: The visualizations are inherently tied to anatomical structures (hemiretinas, neuron-axon relationships)
- **Multi-Scale Analysis**: CARE can be applied at different network depths to understand hierarchical attention patterns
- **Interpretable Heatmaps**: Generates normalized, interpretable heatmaps showing which anatomical regions the model focuses on

### 3. SSL (Self-Supervised Learning) Fine-tuning with Multi-Task Consistency

A novel self-supervised fine-tuning approach that enforces consistency between Grad-CAM and CARE visualizations:

- **Multi-Task Consistency Loss**: Combines classification loss with consistency loss between Grad-CAM and CARE attention maps
- **Flexible Loss Functions**: Supports multiple loss functions (MSE, KL-divergence, SSIM, Pearson correlation) for attention consistency
- **Weighted Training**: Configurable weighting between classification and consistency objectives
- **Improved Generalization**: The consistency constraint helps the model learn more robust and interpretable attention patterns

## AICNet3D Architecture

The AICNet3D architecture combines traditional 3D convolutional layers with the novel cross-attention mechanisms. The model supports configurable attention mechanisms that can be inserted at specific layer indices.

### Key Features:
- **3D Convolutional Backbone**: Standard 3D CNN layers for feature extraction
- **Configurable Attention**: Cross-attention blocks can be inserted at specified layer indices
- **Dual Output**: Returns both classification predictions and CARE attention maps
- **Grad-CAM Integration**: Built-in Grad-CAM computation for comparison with CARE
- **Flexible Architecture**: Supports different attention types and layer configurations

## Environment Setup

```bash
conda env create -f environment.yml
conda activate DL
```

## Data Preparation

1. Organize your OCT data in the following structure:
```
your_data_directory/
├── glaucoma_data/
│   └── *.npy files (3D OCT volumes)
└── non_glaucoma_data/
    └── *.npy files (3D OCT volumes)
```

2. Update the configuration files (`configs/train.yaml` and `configs/ssl_finetune.yaml`) with your data paths:
```yaml
dataset:
  glaucoma_dir: /path/to/your/glaucoma/data
  non_glaucoma_dir: /path/to/your/non_glaucoma/data
```

## Training

### Standard Training
```bash
./train.sh
```

### SSL Fine-tuning
```bash
./ssl_finetune.sh
```

## Configuration

Edit the YAML configuration files in `configs/` to customize model parameters, dataset paths, and training settings.

### Key Configuration Parameters:

```yaml
model_params:
  att_type: "HemiretinalNeuronAxon"  # Attention mechanism type
  att_ind: [2, 4]                    # Layer indices for attention blocks
  CARE_layer_num: 4                  # Layer for CARE visualization
  gradcam_layer_num: 5               # Layer for Grad-CAM computation
  num_conv_layers: 5                 # Number of convolutional layers

experiment:
  use_ssl: true                      # Enable SSL fine-tuning
  ssl_loss_name: "MSE"               # SSL loss function
  ssl_weight: 0.75                   # Weight for SSL loss
```

### Available Attention Types:
- `"Hemiretinal"`: Superior-inferior hemiretinal attention
- `"NeuronAxon"`: Macular-neural attention
- `"HemiretinalNeuronAxon"`: Combined attention
- `"EPA"`: Efficient Paired Attention

### Available SSL Loss Functions:
- `"MSE"`: Mean Squared Error
- `"KL"`: Kullback-Leibler Divergence
- `"SSIM"`: Structural Similarity Index
- `"Pearson"`: Pearson Correlation Loss

## Visualization and Analysis

### CARE vs Grad-CAM Comparison
The framework provides tools to compare CARE and Grad-CAM visualizations:

- **CARE Heatmaps**: Direct attention extraction from cross-attention layers
- **Grad-CAM Heatmaps**: Gradient-based attention visualization
- **Consistency Analysis**: Quantitative comparison between the two methods
- **Overlay Visualization**: Combined visualization of attention maps with original OCT images

### Attention Analysis Tools
Located in `overlap_calculations/`:
- `calc_metrics.py`: Quantitative analysis of attention map overlap with anatomical masks
- `make_mask_overlay.py`: Create overlay visualizations of attention maps with anatomical structures

## Model Outputs

The trained model provides:
1. **Classification Predictions**: Binary glaucoma classification (glaucoma/non-glaucoma)
2. **CARE Attention Maps**: Anatomically-informed attention visualizations
3. **Grad-CAM Heatmaps**: Gradient-based attention visualizations
4. **Training Metrics**: Accuracy, sensitivity, specificity, AUROC, F1-score

## File Structure

```
├── configs/                    # Configuration files
│   ├── train.yaml             # Standard training configuration
│   └── ssl_finetune.yaml      # SSL fine-tuning configuration
├── overlap_calculations/       # Attention analysis tools
│   ├── calc_metrics.py        # Quantitative attention analysis
│   └── make_mask_overlay.py   # Visualization tools
├── model.py                   # AICNet3D architecture
├── transformer_block.py       # Cross-attention mechanisms
├── train.py                   # Standard training script
├── ssl_finetune.py           # SSL fine-tuning script
├── OCT_dataloader.py         # Data loading utilities
├── losses_eval.py            # Custom loss functions
├── transforms.py             # Data augmentation
└── environment.yml           # Conda environment specification
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{melba:2025:018:kenia,
    title = "AI-CNet3D: An Anatomically-Informed Cross-Attention Network with Multi-Task Consistency Fine-tuning for 3D Glaucoma Classification",
    author = "Kenia, Roshan and Li, Anfei and Srivastava, Rishabh and Thakoor, Kaveri A.",
    journal = "Machine Learning for Biomedical Imaging",
    volume = "3",
    issue = "August 2025 issue",
    year = "2025",
    pages = "402--424",
    issn = "2766-905X",
    doi = "https://doi.org/10.59275/j.melba.2025-8d4c",
    url = "https://melba-journal.org/"
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on PyTorch and MONAI frameworks
- Inspired by anatomical knowledge of optic nerve head structure
- Cross-attention mechanisms based on transformer architectures
