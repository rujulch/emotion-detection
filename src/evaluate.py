"""
Evaluation script for Facial Emotion Recognition models.
Generates confusion matrix, classification report, and per-class metrics.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

from config import (
    setup_gpu, CNN_MODEL_PATH, EFFICIENTNET_MODEL_PATH,
    EMOTION_LABELS, NUM_CLASSES, BATCH_SIZE
)
from data_loader import get_data_generators_cnn, get_data_generators_efficientnet


def load_model(model_path):
    """Load a trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def get_predictions(model, generator):
    """
    Get predictions for all samples in a generator.
    
    Returns:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
    """
    # Reset generator
    generator.reset()
    
    # Get predictions
    print("Generating predictions...")
    y_probs = model.predict(generator, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    
    # Get true labels
    y_true = generator.classes
    
    return y_true, y_pred, y_probs


def plot_confusion_matrix(y_true, y_pred, labels, save_path=None, normalize=True):
    """
    Plot and optionally save a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        save_path: Path to save the figure (optional)
        normalize: Whether to normalize the confusion matrix
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_per_class_accuracy(y_true, y_pred, labels, save_path=None):
    """
    Plot per-class accuracy as a bar chart.
    """
    # Calculate per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Sort by accuracy
    sorted_indices = np.argsort(per_class_acc)
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_acc = per_class_acc[sorted_indices]
    
    # Color by performance
    colors = ['#ff6b6b' if acc < 0.5 else '#ffd93d' if acc < 0.7 else '#6bcb77' 
              for acc in sorted_acc]
    
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_labels, sorted_acc, color=colors, edgecolor='black')
    
    # Add value labels
    for bar, acc in zip(bars, sorted_acc):
        plt.text(acc + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{acc*100:.1f}%', va='center', fontsize=10)
    
    plt.xlim(0, 1.1)
    plt.xlabel('Accuracy', fontsize=12)
    plt.ylabel('Emotion', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    plt.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-class accuracy plot saved to: {save_path}")
    
    plt.show()


def print_classification_report(y_true, y_pred, labels):
    """Print a formatted classification report."""
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    print(report)
    return report


def evaluate_model(model_type='cnn', save_plots=True):
    """
    Evaluate a trained model on the test set.
    
    Args:
        model_type: 'cnn' or 'efficientnet'
        save_plots: Whether to save plots to disk
    """
    print("\n" + "="*60)
    print(f"EVALUATING {model_type.upper()} MODEL")
    print("="*60)
    
    # Setup GPU
    setup_gpu()
    
    # Load model and data based on type
    if model_type == 'cnn':
        model = load_model(CNN_MODEL_PATH)
        _, _, test_gen = get_data_generators_cnn(batch_size=BATCH_SIZE)
        plots_dir = os.path.join(os.path.dirname(CNN_MODEL_PATH), 'plots_cnn')
    else:
        model = load_model(EFFICIENTNET_MODEL_PATH)
        _, _, test_gen = get_data_generators_efficientnet(batch_size=BATCH_SIZE)
        plots_dir = os.path.join(os.path.dirname(EFFICIENTNET_MODEL_PATH), 'plots_efficientnet')
    
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get label names in correct order
    label_names = [EMOTION_LABELS[i] for i in range(NUM_CLASSES)]
    
    # Get predictions
    y_true, y_pred, y_probs = get_predictions(model, test_gen)
    
    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    print_classification_report(y_true, y_pred, label_names)
    
    # Confusion matrices
    if save_plots:
        # Normalized confusion matrix
        plot_confusion_matrix(
            y_true, y_pred, label_names,
            save_path=os.path.join(plots_dir, 'confusion_matrix_normalized.png'),
            normalize=True
        )
        
        # Raw confusion matrix
        plot_confusion_matrix(
            y_true, y_pred, label_names,
            save_path=os.path.join(plots_dir, 'confusion_matrix_raw.png'),
            normalize=False
        )
        
        # Per-class accuracy
        plot_per_class_accuracy(
            y_true, y_pred, label_names,
            save_path=os.path.join(plots_dir, 'per_class_accuracy.png')
        )
    else:
        plot_confusion_matrix(y_true, y_pred, label_names, normalize=True)
        plot_per_class_accuracy(y_true, y_pred, label_names)
    
    # Find most confused pairs
    print("\n" + "="*60)
    print("MOST CONFUSED EMOTION PAIRS")
    print("="*60)
    cm = confusion_matrix(y_true, y_pred)
    # Zero out diagonal
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    
    # Find top 5 confusion pairs
    flat_indices = np.argsort(cm_no_diag.flatten())[::-1][:5]
    for idx in flat_indices:
        i, j = divmod(idx, NUM_CLASSES)
        if cm_no_diag[i, j] > 0:
            print(f"  {label_names[i]} → {label_names[j]}: {cm_no_diag[i, j]} samples ({cm_no_diag[i, j]/cm[i].sum()*100:.1f}%)")
    
    return accuracy, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description='Evaluate Emotion Recognition Models')
    parser.add_argument('--model', type=str, choices=['cnn', 'efficientnet', 'both'],
                       default='cnn', help='Model to evaluate')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save plots to disk')
    
    args = parser.parse_args()
    
    save_plots = not args.no_save
    
    if args.model == 'cnn' or args.model == 'both':
        try:
            evaluate_model('cnn', save_plots=save_plots)
        except FileNotFoundError as e:
            print(f"Skipping CNN evaluation: {e}")
    
    if args.model == 'efficientnet' or args.model == 'both':
        try:
            evaluate_model('efficientnet', save_plots=save_plots)
        except FileNotFoundError as e:
            print(f"Skipping EfficientNet evaluation: {e}")


if __name__ == "__main__":
    main()


