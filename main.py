"""
MNIST Digit Classification using Neural Network

A simple educational project demonstrating:
1. Data loading and preprocessing
2. Building a Multi-Layer Perceptron (MLP) model
3. Training and evaluation
4. Making predictions

Author: Educational Project
Purpose: Teaching Neural Network fundamentals for students
"""

import argparse
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Import our modules
from src.data_loader import load_and_preprocess_data
from src.model import create_and_compile_model
from src.train import train_model
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_training_history
from src.visualize import visualize_sample_data, plot_class_distribution
from src.predict import predict_sample_images


def main():
    """
    Main function to run the complete MNIST classification pipeline
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MNIST Digit Classification')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    args = parser.parse_args()
    
    print("="*60)
    print("MNIST DIGIT CLASSIFICATION PROJECT")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\n" + "="*60)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("="*60)
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()
    
    # Step 2: Visualize sample data
    print("\n" + "="*60)
    print("STEP 2: VISUALIZING DATA")
    print("="*60)
    visualize_sample_data(X_train, y_train)
    plot_class_distribution(y_train)
    
    # Step 3: Create model
    print("\n" + "="*60)
    print("STEP 3: CREATING MODEL")
    print("="*60)
    model = create_and_compile_model(learning_rate=args.learning_rate)
    
    # Step 4: Train model
    print("\n" + "="*60)
    print("STEP 4: TRAINING MODEL")
    print("="*60)
    history = train_model(model, X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    
    # Step 5: Evaluate model
    print("\n" + "="*60)
    print("STEP 5: EVALUATING MODEL")
    print("="*60)
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Step 6: Generate visualizations
    print("\n" + "="*60)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("="*60)
    plot_training_history(history)
    plot_confusion_matrix(model, X_test, y_test)
    predict_sample_images(model, X_test, y_test)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")
    print(f"\nModel saved to: models/mnist_model.keras")
    print(f"Visualizations saved to: outputs/")
    print("\nGenerated files:")
    print("  - outputs/sample_images.png")
    print("  - outputs/class_distribution.png")
    print("  - outputs/training_history.png")
    print("  - outputs/confusion_matrix.png")
    print("  - outputs/sample_predictions.png")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
