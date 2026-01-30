
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_sample_data(X_train, y_train, num_samples=10):
    """
    Visualize random samples from training data
    
    Args:
        X_train: Training images (flattened)
        y_train: Training labels (one-hot encoded)
        num_samples: Number of samples to display
    """
    print("\nVisualizing sample training data...")
    
    # Get random samples
    indices = np.random.choice(len(X_train), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # Reshape back to 28x28 for display
        img = X_train[idx].reshape(28, 28)
        label = np.argmax(y_train[idx])
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/sample_images.png', dpi=300, bbox_inches='tight')
    print("Sample images saved to 'outputs/sample_images.png'")
    plt.close()


def plot_class_distribution(y_train):
    """
    Plot distribution of classes in training data
    
    Args:
        y_train: Training labels (one-hot encoded)
    """
    print("\nPlotting class distribution...")
    
    # Convert one-hot to class labels
    labels = np.argmax(y_train, axis=1)
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts, color='steelblue')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.title('Class Distribution in Training Data')
    plt.xticks(unique)
    plt.grid(axis='y', alpha=0.3)
    
    for i, count in enumerate(counts):
        plt.text(i, count + 100, str(count), ha='center', va='bottom')
    
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/class_distribution.png', dpi=300, bbox_inches='tight')
    print("Class distribution saved to 'outputs/class_distribution.png'")
    plt.close()
