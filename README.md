# Alzheimer MRI Classification (ResNet18 + Grad-CAM)

This repository implements a deep learning pipeline for classifying stages of Alzheimer’s disease from structural MRI images using a ResNet18 model with transfer learning.  
The project includes full data preprocessing, training, evaluation, and model interpretability via Grad-CAM.

---

## Overview

This project performs supervised image classification on MRI slices labeled into four stages of dementia:

- NonDemented  
- VeryMildDemented  
- MildDemented  
- ModerateDemented  

The workflow includes:

- Dataset preprocessing and train/validation/test splitting  
- ResNet18 training with class imbalance handling  
- Evaluation using accuracy, confusion matrix, classification report, and ROC-AUC  
- Grad-CAM visualization for model interpretability  
- Optional Gradio interface for uploading MRI images and generating predictions  

---

## Dataset

**Source:**  
Augmented Alzheimer MRI Dataset (Kaggle)  
https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset

The dataset consists of axial MRI slices classified into four categories.  
Images were split into train/validation/test sets in an 80/10/10 ratio.

---

## Model Architecture

- ResNet18 pretrained on ImageNet  
- Final fully connected layer modified for four-class classification  
- CrossEntropyLoss with class weights to address data imbalance  
- Adam optimizer (learning rate: 1e-4)  
- Learning rate scheduler (ReduceLROnPlateau)  

---

## Training Pipeline

1. Load dataset and apply image transforms  
2. Create DataLoaders for training, validation, and testing  
3. Train ResNet18 for several epochs  
4. Track loss and accuracy curves  
5. Save best model checkpoint based on validation loss  

---

## Evaluation Metrics

The model is evaluated on the held-out test set using:
- Accuracy
- Confusion Matrix
- Precision, Recall, and F1-score
- Multi-class ROC curves and AUC scores

---

## Grad-CAM Explainability

Grad-CAM visualizations are used to highlight important brain regions contributing to the model’s prediction.
These visualizations help assess whether the model focuses on anatomically meaningful features.

Example:

---

## Future Work
