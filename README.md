# 🔢 MNIST Digit Classification - Simple Neural Network

A beginner-friendly educational project for learning neural networks with the MNIST dataset.

## 📚 What You'll Learn

- How to load and preprocess image data
- Building a simple neural network (Multi-Layer Perceptron)
- Training a model with TensorFlow/Keras
- Evaluating model performance
- Making predictions on new images

## 🚀 Quick Start

### 1. Install Requirements

```bash
cd MNIST_CNN_Classification
pip install -r requirements.txt
```

### 2. Run the Project

```bash
python3 main.py
```

That's it! The program will:
- ✅ Load 60,000 training images
- ✅ Train a neural network (5 epochs, ~1 minute)
- ✅ Test on 10,000 images
- ✅ Generate visualizations
- ✅ Save the trained model

### 3. Expected Results

- **Test Accuracy**: ~89-90%
- **Training Time**: ~1 minute on CPU
- **Model Size**: ~850 KB

## 📁 Project Structure

```
MNIST_CNN_Classification/
├── src/
│   ├── data_loader.py      # Load and preprocess MNIST data
│   ├── model.py             # Define neural network architecture
│   ├── train.py             # Train the model
│   ├── evaluate.py          # Evaluate and visualize results
│   ├── predict.py           # Make predictions
│   └── visualize.py         # Create visualizations
├── models/                  # Saved trained models
├── outputs/                 # Generated visualizations
├── main.py                  # Main script to run everything
└── requirements.txt         # Python dependencies
```

## 🧠 Model Architecture

Simple 3-layer neural network:

```
Input (784 pixels)
    ↓
Dense Layer (256 neurons, ReLU)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Output Layer (10 neurons, Softmax)
```

**Total Parameters**: 218,058

## 📊 Generated Outputs

After running, check the `outputs/` folder:

1. **sample_images.png** - Random training samples
2. **class_distribution.png** - How many of each digit
3. **training_history.png** - Accuracy and loss over time
4. **confusion_matrix.png** - Which digits get confused
5. **sample_predictions.png** - Test predictions with correct/incorrect labels

## 🎯 How It Works

### Step 1: Load Data
```python
from src.data_loader import load_and_preprocess_data
(X_train, y_train), (X_test, y_test) = load_and_preprocess_data()
```

### Step 2: Create Model
```python
from src.model import create_and_compile_model
model = create_and_compile_model()
```

### Step 3: Train
```python
from src.train import train_model
history = train_model(model, X_train, y_train, epochs=5)
```

### Step 4: Evaluate
```python
from src.evaluate import evaluate_model
accuracy = evaluate_model(model, X_test, y_test)
```

## ⚙️ Customization

Train with different parameters:

```bash
# More epochs for better accuracy
python3 main.py --epochs 10

# Different batch size
python3 main.py --batch-size 64

# Different learning rate
python3 main.py --learning-rate 0.01
```

## 🔍 Understanding the Code

### Data Preprocessing
- Images are flattened from 28×28 to 784 pixels
- Pixel values normalized to [0, 1] range
- Labels converted to one-hot encoding

### Model Training
- **Optimizer**: SGD (Stochastic Gradient Descent)
- **Loss Function**: Categorical Crossentropy
- **Activation**: ReLU for hidden layers, Softmax for output

### Evaluation
- Accuracy on 10,000 test images
- Confusion matrix shows prediction patterns
- Training history shows learning progress

## 📝 For Students

This project is designed to be:
- **Simple**: Easy to understand code with comments
- **Complete**: Full pipeline from data to predictions
- **Visual**: Lots of graphs to see what's happening
- **Educational**: Learn by running and modifying

### Try These Experiments:

1. **Change the architecture**: Add more layers or neurons
2. **Adjust learning rate**: See how it affects training
3. **Train longer**: Use more epochs
4. **Modify batch size**: Observe the impact on training

## 🎓 Key Concepts

### Neural Network
A network of artificial neurons that learns patterns from data.

### ReLU Activation
Rectified Linear Unit - outputs input if positive, else zero.

### Softmax
Converts outputs to probabilities that sum to 1.

### One-Hot Encoding
Converts labels to binary vectors (e.g., 3 → [0,0,0,1,0,0,0,0,0,0]).

### Categorical Crossentropy
Loss function for multi-class classification.

## 📚 Additional Resources

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with the code
- Try different architectures
- Add new features
- Share your results

---

**Happy Learning! 🚀**

Made for students learning machine learning and neural networks.