
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD

def create_and_compile_model(learning_rate=0.001):
    """
    Create a simple Multi-Layer Perceptron (MLP) model for MNIST classification
    
    Architecture:
    - Input: 784 pixels (28x28 flattened)
    - Hidden Layer 1: 256 neurons with ReLU activation
    - Hidden Layer 2: 64 neurons with ReLU activation
    - Output Layer: 10 neurons with Softmax activation (for 10 digits)
    
    Args:
        learning_rate: Learning rate for SGD optimizer
        
    Returns:
        Compiled Keras model
    """
    print("Creating model...")
    
    model = Sequential()
    
    # Input layer + First hidden layer (256 neurons)
    model.add(Dense(256, input_dim=784, activation='relu'))
    
    # Second hidden layer (64 neurons)
    model.add(Dense(64, activation='relu'))
    
    # Output layer (10 classes)
    model.add(Dense(10, activation='softmax'))
    
    # Compile model
    opt = SGD(learning_rate=learning_rate)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    return model
