# CAPTCHA Recognition using Deep Learning

A deep learning-based CAPTCHA recognition system built using PyTorch. The project generates synthetic CAPTCHA images, prepares character-based training data, trains a convolutional neural network (CNN), evaluates character-level performance, and predicts CAPTCHA characters from unseen images.

## 🚀 Project Overview

This project implements an end-to-end CAPTCHA character recognition pipeline:

1. Generate 50,000 synthetic CAPTCHA images.
2. Prepare character-based training data.
3. Preprocess images into grayscale 32 × 32 inputs.
4. Train a CNN using PyTorch.
5. Evaluate the trained model using test accuracy, classification report, and confusion matrix.
6. Segment characters from a CAPTCHA image.
7. Predict individual characters and combine them into the final CAPTCHA text.

## ✨ Key Features

- Generated **50,000 synthetic CAPTCHA images**
- CAPTCHA images of **250 × 70 pixels**
- **5-character** CAPTCHA strings
- Digits, uppercase letters, and lowercase letters
- Random RGB backgrounds and character colors
- Noise lines and noise dots
- **32 × 32 grayscale** model inputs
- RandomAffine and RandomPerspective augmentation
- CNN with Batch Normalization, ReLU, MaxPooling, Dropout, and Adaptive Average Pooling
- **AdamW optimizer**
- ReduceLROnPlateau learning-rate scheduler
- 80/20 training-validation split
- Character segmentation using OpenCV
- Classification report and confusion matrix
- Random test prediction visualization
- Character-level confidence scores

## 🧠 Methodology

```text
Synthetic CAPTCHA Generation
          ↓
Character-based Dataset Preparation
          ↓
Image Preprocessing
          ↓
Data Augmentation
          ↓
CNN Model Training
          ↓
Model Evaluation
          ↓
Character Segmentation
          ↓
Character Prediction
          ↓
Final CAPTCHA Text
Input: 1 × 32 × 32
        ↓
Conv2D: 1 → 32
        ↓
Conv2D: 32 → 32
        ↓
MaxPool + Dropout
        ↓
Conv2D: 32 → 64
        ↓
Conv2D: 64 → 64
        ↓
MaxPool + Dropout
        ↓
Conv2D: 64 → 128
        ↓
Conv2D: 128 → 128
        ↓
MaxPool + Dropout
        ↓
Conv2D: 128 → 256
        ↓
Adaptive Average Pooling
        ↓
Fully Connected Layers
        ↓
Output Classes
captcha_solver/
│
├── captcha_solver.py
├── evaluate.py
├── generate_captcha.py
├── predict.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── captcha_cnn.py
│
├── utils/
│   ├── labels.py
│   ├── preprocessing.py
│   └── segmentation.py
│
├── data/
│   ├── captcha/
│   └── train/
│
└── results/
    ├── confusion_matrix.png
    └── random_predictions.png