"""
Utility functions for Facial Emotion Recognition project.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Keras training history object
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


def visualize_augmentations(generator, num_samples=5):
    """
    Visualize data augmentation by showing original and augmented samples.
    
    Args:
        generator: Keras data generator with augmentation
        num_samples: Number of samples to visualize
    """
    # Get a batch
    batch_x, batch_y = next(generator)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i in range(min(num_samples, len(batch_x))):
        img = batch_x[i]
        label_idx = np.argmax(batch_y[i])
        
        # Handle grayscale vs RGB
        if img.shape[-1] == 1:
            axes[i].imshow(img.squeeze(), cmap='gray')
        else:
            axes[i].imshow(img)
        
        axes[i].set_title(f"Class: {label_idx}")
        axes[i].axis('off')
    
    plt.suptitle('Augmented Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def count_dataset_samples(directory):
    """
    Count samples per class in a directory.
    
    Args:
        directory: Path to dataset directory with class subfolders
    
    Returns:
        Dictionary with class names and sample counts
    """
    counts = {}
    
    for class_name in sorted(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            num_samples = len([f for f in os.listdir(class_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            counts[class_name] = num_samples
    
    return counts


def print_dataset_summary(train_dir, test_dir):
    """Print a summary of the dataset."""
    train_counts = count_dataset_samples(train_dir)
    test_counts = count_dataset_samples(test_dir)
    
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    
    print("\n{:<12} {:>10} {:>10} {:>10}".format("Emotion", "Train", "Test", "Total"))
    print("-"*42)
    
    total_train = 0
    total_test = 0
    
    for emotion in train_counts:
        train = train_counts.get(emotion, 0)
        test = test_counts.get(emotion, 0)
        total_train += train
        total_test += test
        print("{:<12} {:>10} {:>10} {:>10}".format(emotion, train, test, train + test))
    
    print("-"*42)
    print("{:<12} {:>10} {:>10} {:>10}".format("TOTAL", total_train, total_test, total_train + total_test))
    print("="*50 + "\n")


if __name__ == "__main__":
    from config import TRAIN_DIR, TEST_DIR
    print_dataset_summary(TRAIN_DIR, TEST_DIR)


