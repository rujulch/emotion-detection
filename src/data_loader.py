"""
Data loading and augmentation for Facial Emotion Recognition.
Implements proper train/validation split with data augmentation.
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import (
    TRAIN_DIR, TEST_DIR, CNN_IMG_SIZE, EFFICIENTNET_IMG_SIZE,
    BATCH_SIZE, VALIDATION_SPLIT, AUGMENTATION_CONFIG, EMOTION_LABELS
)


def get_data_generators_cnn(batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT):
    """
    Create data generators for the Custom CNN model (48x48 grayscale).
    
    Returns:
        train_generator: Training data with augmentation
        val_generator: Validation data (no augmentation)
        test_generator: Test data (no augmentation)
    """
    # Training data generator WITH augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        **AUGMENTATION_CONFIG
    )
    
    # Validation and test data generator WITHOUT augmentation
    val_test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Test data generator (no validation split needed)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator (from training subset)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(CNN_IMG_SIZE, CNN_IMG_SIZE),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator (from validation subset of training data)
    val_generator = val_test_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(CNN_IMG_SIZE, CNN_IMG_SIZE),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Test generator (completely separate test set)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(CNN_IMG_SIZE, CNN_IMG_SIZE),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    # Print dataset info
    print("\n" + "="*50)
    print("CNN Data Loader (48x48 Grayscale)")
    print("="*50)
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Test samples: {test_generator.samples}")
    print(f"Class indices: {train_generator.class_indices}")
    print("="*50 + "\n")
    
    return train_generator, val_generator, test_generator


def get_data_generators_efficientnet(batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT):
    """
    Create data generators for EfficientNet model (224x224 RGB).
    
    Returns:
        train_generator: Training data with augmentation
        val_generator: Validation data (no augmentation)
        test_generator: Test data (no augmentation)
    """
    # Training data generator WITH augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        **AUGMENTATION_CONFIG
    )
    
    # Validation and test data generator WITHOUT augmentation
    val_test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_test_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Test generator
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=False
    )
    
    # Print dataset info
    print("\n" + "="*50)
    print("EfficientNet Data Loader (224x224 RGB)")
    print("="*50)
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Test samples: {test_generator.samples}")
    print(f"Class indices: {train_generator.class_indices}")
    print("="*50 + "\n")
    
    return train_generator, val_generator, test_generator


def get_class_distribution(generator):
    """Get the class distribution from a data generator."""
    classes = generator.classes
    unique, counts = np.unique(classes, return_counts=True)
    distribution = dict(zip([EMOTION_LABELS[i] for i in unique], counts))
    return distribution


def compute_class_weights_from_generator(generator):
    """
    Compute class weights based on actual class distribution in the generator.
    Uses the formula: weight = n_samples / (n_classes * n_samples_for_class)
    """
    classes = generator.classes
    unique, counts = np.unique(classes, return_counts=True)
    n_samples = len(classes)
    n_classes = len(unique)
    
    weights = {}
    for cls, count in zip(unique, counts):
        weights[cls] = n_samples / (n_classes * count)
    
    return weights


if __name__ == "__main__":
    # Test the data loaders
    print("Testing CNN data loader...")
    train_gen, val_gen, test_gen = get_data_generators_cnn()
    
    print("\nClass distribution in training set:")
    dist = get_class_distribution(train_gen)
    for emotion, count in dist.items():
        print(f"  {emotion}: {count}")
    
    print("\nComputed class weights:")
    weights = compute_class_weights_from_generator(train_gen)
    for cls, weight in weights.items():
        print(f"  {EMOTION_LABELS[cls]}: {weight:.2f}")
    
    # Test a batch
    print("\nTesting batch loading...")
    batch_x, batch_y = next(train_gen)
    print(f"Batch X shape: {batch_x.shape}")
    print(f"Batch Y shape: {batch_y.shape}")
    print(f"X value range: [{batch_x.min():.3f}, {batch_x.max():.3f}]")


