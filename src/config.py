"""
Configuration file for Facial Emotion Recognition project.
Contains all hyperparameters, paths, and settings.
"""

import os
import tensorflow as tf

# =============================================================================
# GPU CONFIGURATION
# =============================================================================
def setup_gpu():
    """Configure GPU memory growth to prevent TF from grabbing all VRAM."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[OK] GPU detected: {gpus[0].name}")
            return True
        except RuntimeError as e:
            print(f"[ERROR] GPU setup error: {e}")
            return False
    else:
        print("[WARNING] No GPU detected. Training will use CPU.")
        print("  Note: TensorFlow 2.11+ requires WSL2 for GPU on Windows.")
        return False

# =============================================================================
# PATHS
# =============================================================================
# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Dataset paths - using the fer2013 dataset folder
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'fer2013 dataset', 'train')
TEST_DIR = os.path.join(PROJECT_ROOT, 'fer2013 dataset', 'test')

# Model save paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'emotion_cnn_best.h5')
EFFICIENTNET_MODEL_PATH = os.path.join(MODELS_DIR, 'emotion_efficientnet_best.h5')

# Logs for TensorBoard
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Haar cascade for face detection
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

# =============================================================================
# EMOTION LABELS (Single Source of Truth)
# =============================================================================
# Order matches the folder names when sorted alphabetically
EMOTION_LABELS = {
    0: 'angry',
    1: 'disgust', 
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

NUM_CLASSES = len(EMOTION_LABELS)

# Reverse mapping for predictions
EMOTION_TO_INDEX = {v: k for k, v in EMOTION_LABELS.items()}

# =============================================================================
# CLASS WEIGHTS (Handle Imbalance)
# =============================================================================
# Based on training set distribution:
# angry: 3995, disgust: 436, fear: 4097, happy: 7215, 
# neutral: 4965, sad: 4830, surprise: 3171
# Weights are inversely proportional to class frequency, normalized to happy (largest)
CLASS_WEIGHTS = {
    0: 1.8,    # angry (3995)
    1: 16.5,   # disgust (436) - severely underrepresented
    2: 1.8,    # fear (4097)
    3: 1.0,    # happy (7215) - baseline
    4: 1.5,    # neutral (4965)
    5: 1.5,    # sad (4830)
    6: 2.3     # surprise (3171)
}

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Custom CNN settings (48x48 grayscale)
CNN_IMG_SIZE = 48
CNN_CHANNELS = 1  # Grayscale

# EfficientNet settings (224x224 RGB)
EFFICIENTNET_IMG_SIZE = 224
EFFICIENTNET_CHANNELS = 3  # RGB

# Training settings
BATCH_SIZE = 64
EPOCHS = 50
INITIAL_LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 1e-7

# Validation split from training data
VALIDATION_SPLIT = 0.2

# Early stopping patience
EARLY_STOPPING_PATIENCE = 10

# Learning rate reduction
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 5

# =============================================================================
# DATA AUGMENTATION SETTINGS
# =============================================================================
AUGMENTATION_CONFIG = {
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip': True,
    'zoom_range': 0.1,
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'nearest'
}

# =============================================================================
# REAL-TIME DETECTION SETTINGS
# =============================================================================
# Temporal smoothing - number of frames to average predictions
SMOOTHING_WINDOW = 5

# Minimum confidence to display prediction
MIN_CONFIDENCE = 0.3

# Face detection settings
FACE_SCALE_FACTOR = 1.3
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (30, 30)

# Display settings
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


if __name__ == "__main__":
    # Test configuration
    print("=" * 50)
    print("Emotion Detection Configuration")
    print("=" * 50)
    setup_gpu()
    print(f"\nTrain directory: {TRAIN_DIR}")
    print(f"Test directory: {TEST_DIR}")
    print(f"Train exists: {os.path.exists(TRAIN_DIR)}")
    print(f"Test exists: {os.path.exists(TEST_DIR)}")
    print(f"\nEmotion labels: {EMOTION_LABELS}")
    print(f"\nClass weights: {CLASS_WEIGHTS}")


