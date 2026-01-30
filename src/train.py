
import os

def train_model(model, X_train, y_train, epochs=5, batch_size=32):
    """
    Train the model on MNIST dataset
    
    Args:
        model: Compiled Keras model
        X_train: Training images
        y_train: Training labels (one-hot encoded)
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Training history object
    """
    print(f"\nTraining model for {epochs} epochs with batch size {batch_size}...")
    
    history = model.fit(
        X_train, 
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.1  # Use 10% of training data for validation
    )
    
    print("\nTraining completed!")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model.save('models/mnist_model.keras')
    print("Model saved to 'models/mnist_model.keras'")
    
    return history
