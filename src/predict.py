
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras

def predict_digit(model, image_path):
    """
    Predict digit from a custom image
    
    Args:
        model: Trained Keras model
        image_path: Path to the image file
        
    Returns:
        Predicted digit and probability distribution
    """
    # Read and preprocess image
    img = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Display the image
    plt.imshow(gray_image, cmap='Greys')
    plt.title('Input Image')
    plt.axis('off')
    plt.show()
    
    # Reshape and normalize
    x = np.expand_dims(gray_image, axis=0)
    x = x.reshape((1, -1))
    x = x.astype('float32') / 255.0
    
    # Predict
    prob = model.predict(x)
    pred = np.argmax(prob, axis=1)
    
    print(f'\nPredicted value is: {pred[0]}')
    print(f'Probability across all numbers: {prob[0]}')
    
    return pred[0], prob[0]


def predict_sample_images(model, X_test, y_test, num_samples=10):
    """
    Predict and display random samples from test set
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels (one-hot encoded)
        num_samples: Number of samples to display
    """
    # Get random samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    # Predict
    predictions = model.predict(X_test[indices])
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[indices], axis=1)
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # Reshape back to 28x28 for display
        img = X_test[indices[i]].reshape(28, 28)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {true_classes[i]}\nPred: {pred_classes[i]}')
        axes[i].axis('off')
        
        # Color code: green if correct, red if wrong
        if pred_classes[i] == true_classes[i]:
            axes[i].spines['bottom'].set_color('green')
            axes[i].spines['top'].set_color('green')
            axes[i].spines['left'].set_color('green')
            axes[i].spines['right'].set_color('green')
        else:
            axes[i].spines['bottom'].set_color('red')
            axes[i].spines['top'].set_color('red')
            axes[i].spines['left'].set_color('red')
            axes[i].spines['right'].set_color('red')
    
    plt.tight_layout()
    plt.savefig('outputs/sample_predictions.png', dpi=300, bbox_inches='tight')
    print("Sample predictions saved to 'outputs/sample_predictions.png'")
    plt.close()
