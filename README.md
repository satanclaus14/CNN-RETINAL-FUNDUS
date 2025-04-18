# AI-Based Early Detection of Diabetic Retinopathy

This repository contains an end-to-end deep learning pipeline to detect **Diabetic Retinopathy (DR)** using retinal fundus images from the [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) dataset.

---

## 🩺 Project Objective

Early detection of Diabetic Retinopathy can prevent severe vision loss. This project uses **Convolutional Neural Networks (CNNs)** with transfer learning to classify retinal images into three classes:

- **0 - No DR**
- **1 - Mild/Moderate DR**
- **2 - Severe/Proliferative DR**

The original 5-class labels from the dataset are reduced to 3-class labels for better generalization and model robustness.

---

## 🗂 Dataset

**Source**: [APTOS 2019 Blindness Detection (Kaggle)](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)

- `train.csv`: Image IDs and diagnosis labels.
- `test.csv`: Image IDs only (no labels provided).
- Images are located in `train_images/` and `test_images/`.

---

## 🔍 Preprocessing Pipeline

Each image undergoes the following enhancements:

- ✅ CLAHE (Contrast Limited Adaptive Histogram Equalization)
- ✅ Vessel Segmentation via Adaptive Thresholding
- ✅ Gamma Correction
- ✅ Augmentation using `Albumentations` (flip, noise, brightness, affine transforms)
- ✅ Normalization using ImageNet mean & std

---

## 🧠 Models Used

Transfer learning applied on the following pre-trained architectures:

- `ResNet50`
- `VGG19`
- `InceptionV3`

Each model uses:
- Global Average Pooling
- Dense Layer (128 ReLU)
- Dropout Regularization
- Softmax Output (3 Classes)

---

## 🔀 Ensemble Strategy

Two ensemble models were evaluated:

1. **VGG19 + InceptionV3**
2. **VGG19 + InceptionV3 + ResNet50**

**Soft voting** was applied to average the prediction probabilities.

---

## 📊 Results

| Model                   | Accuracy | ROC AUC (Macro) | F1 Score | Precision | Recall |
|------------------------|----------|------------------|----------|-----------|--------|
| VGG19                  | 0.820    | 0.921            | 0.600    | 0.713     | 0.635  |
| InceptionV3            | 0.819    | 0.910            | 0.660    | 0.767     | 0.664  |
| ResNet50               | **0.844**| **0.940**        | **0.767**| 0.792     | 0.754  |
| Ensemble (VGG+Incep)   | 0.827    | 0.926            | 0.628    | 0.784     | 0.650  |
| Ensemble (All 3)       | 0.839    | 0.939            | 0.709    | **0.809** | **0.700** |

> 🔍 **Best individual model**: ResNet50  
> 🤝 **Best ensemble**: VGG19 + InceptionV3 + ResNet50

---

## 📈 Visualizations

- ROC Curves for each class
- Confusion Matrix
- Class Distribution Bar Plot
- Sample Augmented Fundus Images

---

## 🛠 Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt

## How to Run
Clone the repository 
git clone https://github.com/yourusername/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection

Download the dataset
./aptos2019-blindness-detection/
├── train.csv
├── test.csv
├── train_images/
└── test_images/


## Youtube

[AI-Based Early Detection of Diabetic Retinopathy](https://youtu.be/EGW5cCgDf3s)
