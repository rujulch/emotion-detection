# Facial Emotion Recognition using Deep Learning

A real-time facial emotion recognition system built with TensorFlow and OpenCV. The system classifies seven core emotions from facial expressions using a custom Convolutional Neural Network trained on the FER2013 dataset.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Details](#technical-details)
- [Limitations and Future Work](#limitations-and-future-work)

## Overview

This project implements an end-to-end pipeline for facial emotion recognition:

1. **Data Preprocessing**: Loading and augmenting the FER2013 dataset with proper train/validation/test splits
2. **Model Training**: Custom CNN architecture with BatchNormalization, Dropout, and class weight balancing
3. **Evaluation**: Comprehensive metrics including confusion matrix and per-class accuracy analysis
4. **Real-time Detection**: Webcam-based emotion detection with temporal smoothing for stable predictions

## Features

- Custom CNN architecture optimized for 48x48 grayscale facial images
- Real data augmentation (rotation, shifts, flips, zoom, brightness)
- Class weight balancing to handle dataset imbalance
- Training callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- TensorBoard integration for training visualization
- Real-time webcam detection with Haar Cascade face detection
- Temporal smoothing to reduce prediction jitter
- Confidence bars and FPS counter in detection UI

## Project Structure

```
emotion-detection/
├── src/
│   ├── config.py              # Configuration (paths, hyperparameters, labels)
│   ├── data_loader.py         # Data loading with augmentation
│   ├── model.py               # CNN and EfficientNet architectures
│   ├── train.py               # Training script with callbacks
│   ├── evaluate.py            # Evaluation with confusion matrix
│   ├── realtime_detector.py   # Webcam-based real-time detection
│   ├── utils.py               # Helper functions
│   ├── haarcascade_frontalface_default.xml  # Face detection cascade
│   └── models/
│       ├── emotion_cnn_best.h5              # Trained model weights
│       └── plots_cnn/                       # Evaluation plots
│           ├── confusion_matrix_normalized.png
│           ├── confusion_matrix_raw.png
│           └── per_class_accuracy.png
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── PROJECT_EXPLANATION.md     # Detailed code explanation
```

### Legacy Files (Not Used)

The following files are from an earlier version of the project and are not part of the current implementation:

- `src/emotions.ipynb` - Original notebook (replaced by modular scripts)
- `src/2nd model.ipynb` - Transfer learning experiment (replaced by model.py)
- `src/Dataset Prepare.ipynb` - Dataset preparation (replaced by data_loader.py)
- `src/FERModelPrototype.h5` - Old model weights
- `src/fer2013.csv` - Original CSV dataset (now using image folders)

## Dataset

This project uses the **FER2013** (Facial Expression Recognition 2013) dataset.

### Dataset Statistics

| Emotion   | Training | Test  | Total  |
|-----------|----------|-------|--------|
| Angry     | 3,995    | 958   | 4,953  |
| Disgust   | 436      | 111   | 547    |
| Fear      | 4,097    | 1,024 | 5,121  |
| Happy     | 7,215    | 1,774 | 8,989  |
| Neutral   | 4,965    | 1,233 | 6,198  |
| Sad       | 4,830    | 1,247 | 6,077  |
| Surprise  | 3,171    | 831   | 4,002  |
| **Total** | 28,709   | 7,178 | 35,887 |

### Data Split Strategy

- **Training**: 80% of training folder (22,967 images)
- **Validation**: 20% of training folder (5,742 images) - used for monitoring overfitting
- **Test**: Separate test folder (7,178 images) - used only for final evaluation

### Obtaining the Dataset

The FER2013 dataset must be downloaded separately and placed in the `fer2013 dataset/` folder with the following structure:

```
fer2013 dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

## Model Architecture

### Custom CNN

The model consists of 4 convolutional blocks followed by a dense classifier:

```
Input: 48x48x1 (grayscale)

Block 1: Conv2D(64) -> BatchNorm -> ReLU -> Conv2D(64) -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.25)
Block 2: Conv2D(128) -> BatchNorm -> ReLU -> Conv2D(128) -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.25)
Block 3: Conv2D(256) -> BatchNorm -> ReLU -> Conv2D(256) -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.25)
Block 4: Conv2D(512) -> BatchNorm -> ReLU -> Conv2D(512) -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.25)

GlobalAveragePooling2D
Dense(256) -> BatchNorm -> ReLU -> Dropout(0.5)
Dense(128) -> BatchNorm -> ReLU -> Dropout(0.5)
Dense(7, softmax)

Total Parameters: 4,858,567
```

### Key Design Decisions

1. **BatchNormalization**: Added after every convolutional layer for faster convergence and regularization
2. **GlobalAveragePooling2D**: Reduces parameters compared to Flatten, provides translation invariance
3. **Dropout**: 0.25 after conv blocks, 0.5 after dense layers to prevent overfitting
4. **Class Weights**: Inversely proportional to class frequency to handle imbalance (especially for "disgust")

## Installation

### Prerequisites

- Python 3.9 or higher
- NVIDIA GPU with CUDA support (optional, but recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the FER2013 dataset and place it in the `fer2013 dataset/` folder.

### GPU Support

- **Windows**: TensorFlow 2.11+ requires WSL2 for GPU support
- **Linux**: Install CUDA Toolkit and cuDNN compatible with your TensorFlow version

## Usage

### Training

Train the CNN model from scratch:

```bash
cd src
python train.py --model cnn --epochs 50
```

Training options:
- `--model`: Model type (`cnn` or `efficientnet`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 64)

### Evaluation

Evaluate the trained model and generate confusion matrix:

```bash
cd src
python evaluate.py
```

This will output:
- Overall test accuracy
- Per-class precision, recall, and F1-score
- Confusion matrix plots (saved to `src/models/plots_cnn/`)
- Most confused emotion pairs

### Real-time Detection

Run the webcam-based emotion detector:

```bash
cd src
python realtime_detector.py
```

Controls:
- Press `q` to quit
- Press `r` to reset temporal smoothing

Detection options:
- `--model`: Model type (`cnn` or `efficientnet`)
- `--camera`: Camera device ID (default: 0)
- `--smoothing`: Temporal smoothing window size (default: 5)

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 65.95% |
| **Training Accuracy** | ~70% |
| **Best Epoch** | 49/50 |

### Per-Class Performance

| Emotion  | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Angry    | 56.0%     | 62.8%  | 59.2%    |
| Disgust  | 63.1%     | 58.6%  | 60.8%    |
| Fear     | 50.5%     | 46.0%  | 48.1%    |
| Happy    | 89.6%     | 84.2%  | 86.8%    |
| Neutral  | 59.2%     | 67.6%  | 63.1%    |
| Sad      | 57.2%     | 49.8%  | 53.3%    |
| Surprise | 71.5%     | 78.0%  | 74.6%    |

### Observations

- **Happy** and **Surprise** achieve the highest accuracy due to distinctive facial features
- **Fear** and **Sad** are most challenging, often confused with each other and neutral
- **Disgust** performs well despite having the fewest samples, thanks to class weighting
- The 65.95% accuracy is competitive with published benchmarks on FER2013 (state-of-the-art: ~73%)

## Technical Details

### Data Augmentation

Applied during training to improve generalization:

```python
- Rotation: +/- 15 degrees
- Width/Height shift: 10%
- Horizontal flip: True
- Zoom: +/- 10%
- Brightness: 80-120%
```

### Training Configuration

- **Optimizer**: Adam with initial learning rate 0.001
- **Loss**: Categorical Cross-Entropy
- **Batch Size**: 64
- **Early Stopping**: Patience of 10 epochs, monitoring validation accuracy
- **Learning Rate Reduction**: Factor 0.5, patience 5 epochs

### Class Weights

To handle class imbalance, inversely proportional weights are applied:

| Class    | Weight |
|----------|--------|
| Angry    | 1.8    |
| Disgust  | 16.5   |
| Fear     | 1.8    |
| Happy    | 1.0    |
| Neutral  | 1.5    |
| Sad      | 1.5    |
| Surprise | 2.3    |

### Real-time Detection

- **Face Detection**: Haar Cascade classifier (loaded once at initialization)
- **Temporal Smoothing**: Averages predictions over last 5 frames
- **Minimum Confidence**: 30% threshold for displaying predictions
- **Preprocessing**: Grayscale conversion, resize to 48x48, normalize to [0,1]

## Limitations and Future Work

### Current Limitations

1. **Dataset Quality**: FER2013 contains noisy labels and low-resolution (48x48) images
2. **Class Imbalance**: "Disgust" has only 547 samples vs. 8,989 for "Happy"
3. **Similar Emotions**: Fear, Sad, and Neutral are inherently difficult to distinguish
4. **Lighting Sensitivity**: Performance degrades in poor lighting conditions

### Potential Improvements

1. **Transfer Learning**: Fine-tune EfficientNet or ResNet pretrained on larger face datasets
2. **Attention Mechanisms**: Add SE-Net or CBAM blocks to focus on discriminative regions
3. **Multi-task Learning**: Combine emotion recognition with face landmark detection
4. **Data Collection**: Augment with additional emotion datasets (AffectNet, RAF-DB)
5. **Ensemble Methods**: Combine predictions from multiple models

## References

1. Goodfellow, I. J., et al. "Challenges in Representation Learning: A report on three machine learning contests." Neural Networks (2013).
2. FER2013 Dataset: https://www.kaggle.com/datasets/msambare/fer2013

## License

This project is for educational purposes.

## Author

Built as a deep learning project demonstrating end-to-end emotion recognition from data preprocessing to real-time deployment.

