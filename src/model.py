"""
Model architectures for Facial Emotion Recognition.
Includes: Improved Custom CNN and EfficientNet Transfer Learning.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from config import (
    CNN_IMG_SIZE, CNN_CHANNELS, EFFICIENTNET_IMG_SIZE, 
    EFFICIENTNET_CHANNELS, NUM_CLASSES, INITIAL_LEARNING_RATE
)


def create_cnn_model(input_shape=None, num_classes=NUM_CLASSES, learning_rate=INITIAL_LEARNING_RATE):
    """
    Create an improved Custom CNN for 48x48 grayscale emotion recognition.
    
    Architecture:
    - 4 Conv blocks with BatchNorm and Dropout
    - GlobalAveragePooling instead of Flatten (reduces parameters)
    - Dense head with BatchNorm
    
    Args:
        input_shape: Tuple of (height, width, channels). Defaults to (48, 48, 1)
        num_classes: Number of emotion classes (default: 7)
        learning_rate: Initial learning rate for Adam optimizer
    
    Returns:
        Compiled Keras model
    """
    if input_shape is None:
        input_shape = (CNN_IMG_SIZE, CNN_IMG_SIZE, CNN_CHANNELS)
    
    model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 3
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 4
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Head
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_efficientnet_model(input_shape=None, num_classes=NUM_CLASSES, 
                               learning_rate=INITIAL_LEARNING_RATE, trainable_base=False):
    """
    Create an EfficientNetB0-based model for emotion recognition.
    
    Uses transfer learning with ImageNet weights.
    
    Args:
        input_shape: Tuple of (height, width, channels). Defaults to (224, 224, 3)
        num_classes: Number of emotion classes (default: 7)
        learning_rate: Initial learning rate for Adam optimizer
        trainable_base: Whether to allow training of base model layers
    
    Returns:
        Compiled Keras model
    """
    if input_shape is None:
        input_shape = (EFFICIENTNET_IMG_SIZE, EFFICIENTNET_IMG_SIZE, EFFICIENTNET_CHANNELS)
    
    # Load EfficientNetB0 with ImageNet weights, without top layers
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = trainable_base
    
    # Build model
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def unfreeze_efficientnet_layers(model, num_layers_to_unfreeze=20, learning_rate=1e-5):
    """
    Unfreeze the top layers of EfficientNet for fine-tuning.
    
    Args:
        model: The EfficientNet model to modify
        num_layers_to_unfreeze: Number of layers from the top to unfreeze
        learning_rate: New (lower) learning rate for fine-tuning
    
    Returns:
        Recompiled model with unfrozen layers
    """
    # Find the base model (EfficientNetB0)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
    
    if base_model is None:
        print("Warning: Could not find base model to unfreeze")
        return model
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze all layers except the last num_layers_to_unfreeze
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"Unfroze {trainable_count} layers for fine-tuning")
    print(f"New learning rate: {learning_rate}")
    
    return model


def get_model_summary(model, model_name="Model"):
    """Print a formatted model summary."""
    print("\n" + "="*60)
    print(f"{model_name} Summary")
    print("="*60)
    model.summary()
    
    trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    print(f"\nTrainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {non_trainable:,}")
    print(f"Total parameters: {trainable + non_trainable:,}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("Testing model creation...")
    
    # Test CNN model
    print("\n1. Creating Custom CNN model...")
    cnn_model = create_cnn_model()
    get_model_summary(cnn_model, "Custom CNN (48x48 Grayscale)")
    
    # Test EfficientNet model
    print("\n2. Creating EfficientNet model...")
    effnet_model = create_efficientnet_model()
    get_model_summary(effnet_model, "EfficientNetB0 (224x224 RGB)")
    
    # Test unfreezing
    print("\n3. Testing layer unfreezing...")
    effnet_model = unfreeze_efficientnet_layers(effnet_model, num_layers_to_unfreeze=20)
    
    print("\n✓ All models created successfully!")


