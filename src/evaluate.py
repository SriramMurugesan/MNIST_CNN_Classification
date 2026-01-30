
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test dataset
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels (one-hot encoded)
        
    Returns:
        Test accuracy
    """
    print("\nEvaluating model on test data...")
    
    scores = model.evaluate(X_test, y_test, verbose=1)
    
    print(f"\nTest Loss: {scores[0]:.4f}")
    print(f"Test Accuracy: {scores[1]*100:.2f}%")
    print(f"Error Rate: {(100 - scores[1]*100):.2f}%")
    
    return scores[1]


def plot_confusion_matrix(model, X_test, y_test):
    """
    Generate and plot confusion matrix
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels (one-hot encoded)
    """
    print("\nGenerating confusion matrix...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to 'outputs/confusion_matrix.png'")
    plt.close()


def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: Training history object from model.fit()
    """
    print("\nGenerating training history plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/training_history.png', dpi=300, bbox_inches='tight')
    print("Training history saved to 'outputs/training_history.png'")
    plt.close()
