# Explainable Multi-Task Transformer Framework for Brain Tumor Classification & Segmentation

## Overview
This repository presents an explainable multi-task deep learning framework for simultaneous **brain tumor classification and segmentation** using MRI images. The proposed architecture combines a shared **BEiT transformer encoder** with an attention-enhanced lightweight **SegFormer decoder** to jointly learn both tasks within a unified framework.

The model integrates:
- **Transformer-based feature extraction (BEiT)**
- **Attention-enhanced segmentation decoder (CBAM)**
- **Deep supervision**
- **Stage-wise training**
- **Explainable AI using Eigen-CAM**

The framework is designed to reduce redundant computation between classification and segmentation while improving feature sharing, segmentation accuracy, and interpretability.

---

# Key Features

- Joint multi-task learning for:
  - Brain tumor classification
  - Tumor segmentation

- Shared transformer encoder:
  - BEiT backbone

- Attention-enhanced decoder:
  - CBAM (Convolutional Block Attention Module)

- Deep supervision for improved convergence

- Two-stage training strategy:
  - Stage-1: Joint classification + segmentation
  - Stage-2: High-resolution segmentation refinement

- Explainability pipeline:
  - Segmentation-guided Eigen-CAM
  - Stage-1 vs Stage-2 attention comparison

- Test-Time Augmentation (TTA)

- ROC-AUC evaluation and visualization

---

# Architecture

## Encoder
- **BEiT Transformer Encoder**
- Extracts hierarchical multi-scale feature representations from MRI images.

## Classification Branch
- Global Average Pooling
- Fully Connected Classification Layer
- Softmax prediction for:
  - No Tumor
  - Glioma
  - Meningioma
  - Pituitary Tumor

## Segmentation Branch
- SegFormer-style lightweight decoder
- Multi-scale feature fusion
- CBAM attention refinement
- Deep supervision auxiliary outputs

## Explainability
- Eigen-CAM generated from segmentation decoder activations
- SVD-based principal activation visualization
- Stage-wise CAM comparison

---

# Datasets

## 1. Brain Tumor MRI Classification Dataset
Used for:
- Classification training
- Classification validation/testing

Dataset:
- https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/1

Classes:
- No Tumor
- Glioma
- Meningioma
- Pituitary

---

## 2. Brain Tumor Segmentation Dataset
Used for:
- Segmentation training
- Segmentation validation/testing

Dataset:
- https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset

Contains:
- MRI images
- Corresponding tumor masks

---

## 3. External Mendeley Dataset
Used only for:
- External generalization testing

Dataset:
- https://data.mendeley.com/datasets/zwr4ntf94j/1

Purpose:
- Evaluate robustness on unseen external MRI samples

---

# Multi-Task Dataset Construction

A unified master dataframe is constructed by combining:
- Classification dataset
- Segmentation dataset

Each row contains:
- Image path
- Classification label
- Segmentation mask path (if available)

This allows:
- Classification-only samples
- Segmentation-only samples
- Multi-task samples

to coexist in the same training pipeline.

Missing labels are ignored dynamically during loss computation.

---

# Training Strategy

## Stage-1: Joint Multi-Task Learning
- Image Size: 224×224
- Tasks:
  - Classification
  - Segmentation

### Loss
\[
L_{total} = L_{cls} + \lambda_{seg} L_{seg}
\]

Where:
- CrossEntropy Loss → classification
- Dice + Focal Loss → segmentation

### Objective
Learn shared tumor-aware representations.

---

## Stage-2: Segmentation Refinement
- Image Size: 384×384
- Segmentation-only fine-tuning
- Tumor-focused sampling
- Improved boundary delineation

### Loss
\[
L_{total} = \lambda_{seg} L_{seg}
\]

### Objective
Improve tumor localization and segmentation precision.

---

# Loss Functions

## CrossEntropy Loss
Used for classification.

Measures the difference between:
- predicted class probabilities
- true tumor class labels

Encourages the network to assign high probability to the correct tumor category.

---

## Dice + Focal Loss
Used for segmentation.

### Dice Loss
Measures overlap between:
- predicted mask
- ground-truth mask

Improves region overlap accuracy.

### Focal Loss
Focuses training on:
- difficult pixels
- small tumor regions

Helps handle class imbalance between tumor and background.

---

# Explainable AI (Eigen-CAM)

The framework integrates segmentation-guided Eigen-CAM for interpretability.

## Process
1. Extract decoder activation maps
2. Reshape feature maps
3. Apply Singular Value Decomposition (SVD)
4. Select dominant principal component
5. Generate attention heatmap

The resulting CAM highlights:
- tumor-focused regions
- model attention behavior
- Stage-1 vs Stage-2 refinement differences

---

# Performance

## Classification
- Accuracy: **99.62%**
- F1-score: **0.9959**

## Segmentation (Stage-2)
- Dice Score: **0.9016**
- IoU: **0.8335**

---

# Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- Albumentations
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

# Repository Structure

```bash
├── notebooks/
│   ├── mtl_code.ipynb
│
├── plots/
│   ├── confusion_matrices/
│   ├── roc_auc/
│   ├── learning_curves/
│
├── checkpoints/
│   ├── best_stage1.pth
│   ├── best_stage2_segmentation.pth
│
├── xai_outputs/
│   ├── eigencam_reports/
│
├── README.md
