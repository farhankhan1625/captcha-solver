# CAPTCHA Recognition using Deep Learning

A deep learning-based CAPTCHA recognition system built using PyTorch. The project generates synthetic CAPTCHA images, preprocesses them, trains a convolutional neural network (CNN), and predicts CAPTCHA characters from unseen images.

## 🚀 Project Overview

This project implements an end-to-end CAPTCHA recognition pipeline:

1. Generate synthetic CAPTCHA images with different fonts, colors, noise, and geometric variations.
2. Preprocess CAPTCHA images into grayscale format.
3. Train a PyTorch CNN for character recognition.
4. Evaluate model performance on unseen CAPTCHA samples.
5. Predict CAPTCHA characters using the trained model.

## ✨ Key Features

- Generated **50,000 synthetic CAPTCHA images**
- **32 × 32 grayscale** image inputs
- Randomized fonts, colors, noise, and geometric variations
- CNN-based deep learning model using **PyTorch**
- Affine and perspective data augmentation
- **AdamW optimizer**
- Learning-rate scheduling
- Separate training, evaluation, generation, and prediction scripts

## 🧠 Methodology

```text
Synthetic CAPTCHA Generation
          ↓
Image Preprocessing
          ↓
Data Augmentation
          ↓
CNN Model Training
          ↓
Model Evaluation
          ↓
CAPTCHA Prediction
