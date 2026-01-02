# XAI for Lung Pathologies: Analyzing the Stability of Vision Transformer Explanations under Error-Prone Conditions

**Research project** focused on explainable AI (XAI) in medical imaging, specifically comparing the stability of explanations from convolutional (ResNet50) and transformer-based (Vision Transformer) models on chest X-ray classification under noisy and perturbed conditions.

### Task
Binary classification of chest X-ray images:
- **Normal** vs **Pneumonia** (used as a proxy for lung pathology detection)

### Dataset
- Source: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle)
- Balanced subset: 670 NORMAL + 670 PNEUMONIA images
- Split (per class):
  - Train: 469 (~70%)
  - Validation: 101 (~15%)
  - Test: 100 (~15%)

The full dataset is **not included** in the repository due to size.  
See `data/README.md` and `data/prepare_dataset.py` for automatic preparation instructions.

### Repository Structure
data/                        #   Dataset preparation (no raw images)

prepare_dataset.py           # Script to download and balance the dataset

README.md                    # Instructions for data reproduction

samples/                     # Few example images (optional)

models/
resnet/                      # ResNet50 weights

resnet50_best.pth            # Best checkpoint (highest val accuracy)

vit/                         # Vision Transformer weights (coming soon)

explainability/
resnet_gradcam/              # Grad-CAM heatmaps on clean + noisy test samples

vit_attention/               # ViT attention maps (coming soon)

noise/                       # Examples of perturbed test images

gaussian_low/

gaussian_medium/

blur_light/

blur_medium/

contrast_low/

contrast_high/

results/                      # Quantitative results and figures

figures/                      # Final plots for the paper

tables/                       # Metrics and stability analysis

notebooks/                    # Jupyter/Colab notebooks (coming soon)

### Key Findings (ResNet50 Baseline)
- Gaussian noise is the most disruptive perturbation, significantly increasing false positives on normal images.
- Increased contrast often corrects false positives by sharpening focus on lung regions.
- Blur has moderate impact, with stronger blur leading to loss of discriminative focus.
- Grad-CAM visualizations reveal how perturbations shift model attention from anatomical lung areas to artifacts or background.

Full ViT comparison and detailed stability analysis coming soon.

### Reproducibility
1. **Dataset**
   ```bash
   # Download original dataset from Kaggle and unzip to chest_xray/chest_xray/
   python data/prepare_dataset.py
Output: lung_data/train | val | test/normal | pneumonia

Model & Visualizations
Load models/resnet/resnet50_best.pth with torchvision ResNet50 (pretrained backbone + fine-tuned FC layer).
Grad-CAM heatmaps: see explainability/resnet_gradcam/


Requirements

Python 3.8+
PyTorch ≥ 2.0
torchvision
pytorch-grad-cam
matplotlib, seaborn, scikit-learn

Bashpip install torch torchvision pytorch-grad-cam matplotlib seaborn scikit-learn
License
MIT License (feel free to use, modify, and cite).
