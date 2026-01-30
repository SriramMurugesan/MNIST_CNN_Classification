
import numpy as np
from tensorflow import keras

def load_and_preprocess_data():
    """
    Load and preprocess MNIST dataset
    
    Returns:
        Tuple of (X_train, y_train), (X_test, y_test)
    """
    print("Loading MNIST dataset...")
    
    # Load MNIST data from Keras
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f'Total no of Images: {X_train.shape[0]}')
    print(f'Size of Image: {X_train.shape[1:]}')
    print(f'Total no of labels: {y_train.shape}')
    
    # Reshape images from 28x28 to 784 (flatten)
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    
    # Convert to float32
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    # Normalize pixel values to range [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    print(f'Reshaped training data: {X_train.shape}')
    print(f'Reshaped test data: {X_test.shape}')
    
    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f'One-hot encoded labels: {y_train.shape}')
    
    return (X_train, y_train), (X_test, y_test)
