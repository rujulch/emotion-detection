"""
Training script for Facial Emotion Recognition models.
Includes callbacks, learning rate scheduling, and model checkpointing.
"""

import os
import argparse
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)

from config import (
    setup_gpu, EPOCHS, BATCH_SIZE, CLASS_WEIGHTS,
    CNN_MODEL_PATH, EFFICIENTNET_MODEL_PATH, LOGS_DIR,
    EARLY_STOPPING_PATIENCE, LR_REDUCE_FACTOR, LR_REDUCE_PATIENCE,
    MIN_LEARNING_RATE
)
from data_loader import get_data_generators_cnn, get_data_generators_efficientnet
from model import (
    create_cnn_model, create_efficientnet_model, 
    unfreeze_efficientnet_layers, get_model_summary
)


def get_callbacks(model_path, model_name="model"):
    """
    Create training callbacks.
    
    Args:
        model_path: Path to save the best model
        model_name: Name for TensorBoard logs
    
    Returns:
        List of Keras callbacks
    """
    # Create timestamped log directory
    log_dir = os.path.join(LOGS_DIR, f"{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    
    callbacks = [
        # Early stopping - stop if val_loss doesn't improve
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        ),
        
        # Save best model based on validation accuracy
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    print(f"\nTensorBoard logs will be saved to: {log_dir}")
    print(f"Best model will be saved to: {model_path}")
    
    return callbacks


def train_cnn(epochs=EPOCHS, batch_size=BATCH_SIZE, use_class_weights=True):
    """
    Train the Custom CNN model.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        use_class_weights: Whether to use class weights for imbalanced data
    
    Returns:
        Trained model and training history
    """
    print("\n" + "="*60)
    print("TRAINING CUSTOM CNN MODEL")
    print("="*60)
    
    # Setup GPU
    setup_gpu()
    
    # Load data
    train_gen, val_gen, test_gen = get_data_generators_cnn(batch_size=batch_size)
    
    # Create model
    model = create_cnn_model()
    get_model_summary(model, "Custom CNN")
    
    # Get callbacks
    callbacks = get_callbacks(CNN_MODEL_PATH, "cnn")
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=CLASS_WEIGHTS if use_class_weights else None,
        verbose=1
    )
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    return model, history


def train_efficientnet(epochs=EPOCHS, batch_size=BATCH_SIZE, 
                       use_class_weights=True, fine_tune=True):
    """
    Train the EfficientNet model with optional fine-tuning.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        use_class_weights: Whether to use class weights
        fine_tune: Whether to fine-tune the base model after initial training
    
    Returns:
        Trained model and training history
    """
    print("\n" + "="*60)
    print("TRAINING EFFICIENTNET MODEL")
    print("="*60)
    
    # Setup GPU
    setup_gpu()
    
    # Load data
    train_gen, val_gen, test_gen = get_data_generators_efficientnet(batch_size=batch_size)
    
    # Create model with frozen base
    model = create_efficientnet_model(trainable_base=False)
    get_model_summary(model, "EfficientNetB0 (Frozen Base)")
    
    # Get callbacks for initial training
    callbacks = get_callbacks(EFFICIENTNET_MODEL_PATH, "efficientnet")
    
    # Initial training with frozen base
    initial_epochs = epochs // 2 if fine_tune else epochs
    
    print(f"\nPhase 1: Training with frozen base ({initial_epochs} epochs)...")
    history = model.fit(
        train_gen,
        epochs=initial_epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=CLASS_WEIGHTS if use_class_weights else None,
        verbose=1
    )
    
    # Fine-tuning phase
    if fine_tune:
        print("\n" + "="*60)
        print("PHASE 2: FINE-TUNING")
        print("="*60)
        
        # Unfreeze top layers
        model = unfreeze_efficientnet_layers(model, num_layers_to_unfreeze=20, learning_rate=1e-5)
        
        # Continue training
        fine_tune_epochs = epochs - initial_epochs
        print(f"\nFine-tuning for {fine_tune_epochs} more epochs...")
        
        history_fine = model.fit(
            train_gen,
            epochs=epochs,
            initial_epoch=initial_epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=CLASS_WEIGHTS if use_class_weights else None,
            verbose=1
        )
        
        # Combine histories
        for key in history.history:
            history.history[key].extend(history_fine.history[key])
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train Emotion Recognition Models')
    parser.add_argument('--model', type=str, choices=['cnn', 'efficientnet', 'both'], 
                       default='cnn', help='Model to train')
    parser.add_argument('--epochs', type=int, default=EPOCHS, 
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--no-class-weights', action='store_true',
                       help='Disable class weights')
    parser.add_argument('--no-fine-tune', action='store_true',
                       help='Disable fine-tuning for EfficientNet')
    
    args = parser.parse_args()
    
    use_class_weights = not args.no_class_weights
    
    if args.model == 'cnn' or args.model == 'both':
        train_cnn(
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_class_weights=use_class_weights
        )
    
    if args.model == 'efficientnet' or args.model == 'both':
        train_efficientnet(
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_class_weights=use_class_weights,
            fine_tune=not args.no_fine_tune
        )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nTo view training progress in TensorBoard, run:")
    print(f"  tensorboard --logdir={LOGS_DIR}")
    print("\nTo evaluate the model, run:")
    print("  python evaluate.py")


if __name__ == "__main__":
    main()


