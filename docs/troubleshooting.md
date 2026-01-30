# Troubleshooting Guide

## 🔧 Common Issues and Solutions

This guide helps you resolve common problems when working with the MNIST CNN classification project.

---

## 📦 Installation Issues

### Problem: `pip install` fails
**Symptoms**:
```
ERROR: Could not find a version that satisfies the requirement tensorflow
```

**Solutions**:
1. **Update pip**:
   ```bash
   pip install --upgrade pip
   ```

2. **Check Python version** (requires Python 3.8+):
   ```bash
   python --version
   ```

3. **Use specific TensorFlow version**:
   ```bash
   pip install tensorflow==2.13.0
   ```

4. **For M1/M2 Macs**:
   ```bash
   pip install tensorflow-macos
   pip install tensorflow-metal
   ```

---

### Problem: Import errors
**Symptoms**:
```python
ModuleNotFoundError: No module named 'tensorflow'
```

**Solutions**:
1. **Verify installation**:
   ```bash
   pip list | grep tensorflow
   ```

2. **Check virtual environment**:
   ```bash
   which python  # Should point to venv
   ```

3. **Reinstall in correct environment**:
   ```bash
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

---

## 🏗️ Model Building Issues

### Problem: Shape mismatch errors
**Symptoms**:
```
ValueError: Input 0 of layer "conv2d" is incompatible with the layer
```

**Solutions**:
1. **Check input shape**:
   ```python
   print(x_train.shape)  # Should be (N, 28, 28, 1)
   ```

2. **Ensure proper reshaping**:
   ```python
   x_train = x_train.reshape(-1, 28, 28, 1)
   ```

3. **Verify data preprocessing**:
   ```python
   from src.data_loader import load_and_preprocess_data
   # This handles reshaping automatically
   ```

---

### Problem: Model won't compile
**Symptoms**:
```
TypeError: 'optimizer' must be an instance of Optimizer
```

**Solutions**:
1. **Use correct optimizer syntax**:
   ```python
   from tensorflow.keras.optimizers import Adam
   model.compile(optimizer=Adam(learning_rate=0.001), ...)
   ```

2. **Check TensorFlow version compatibility**:
   ```python
   import tensorflow as tf
   print(tf.__version__)
   ```

---

## 🎓 Training Issues

### Problem: Training is very slow
**Symptoms**: Each epoch takes >5 minutes on CPU

**Solutions**:
1. **Reduce batch size** (uses less memory):
   ```python
   python main.py --batch-size 16
   ```

2. **Reduce epochs for testing**:
   ```python
   python main.py --epochs 3
   ```

3. **Use GPU if available**:
   ```python
   import tensorflow as tf
   print("GPUs:", tf.config.list_physical_devices('GPU'))
   ```

4. **Enable mixed precision** (for GPU):
   ```python
   from tensorflow.keras import mixed_precision
   mixed_precision.set_global_policy('mixed_float16')
   ```

---

### Problem: Loss is NaN
**Symptoms**:
```
Epoch 1/10
loss: nan - accuracy: 0.0000
```

**Solutions**:
1. **Check data normalization**:
   ```python
   print(x_train.min(), x_train.max())  # Should be 0.0, 1.0
   ```

2. **Reduce learning rate**:
   ```python
   python main.py --learning-rate 0.0001
   ```

3. **Check for inf/nan in data**:
   ```python
   import numpy as np
   print(np.isnan(x_train).any())  # Should be False
   ```

4. **Use gradient clipping**:
   ```python
   optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
   ```

---

### Problem: Model not learning (accuracy stuck)
**Symptoms**: Accuracy stays around 10% (random guessing)

**Solutions**:
1. **Verify labels are correct**:
   ```python
   print(y_train[:10])  # Should be integers 0-9
   ```

2. **Check loss function matches labels**:
   ```python
   # For integer labels: use sparse_categorical_crossentropy
   # For one-hot labels: use categorical_crossentropy
   ```

3. **Increase learning rate**:
   ```python
   python main.py --learning-rate 0.01
   ```

4. **Check model architecture**:
   ```python
   model.summary()  # Verify layers are connected
   ```

---

### Problem: Overfitting
**Symptoms**: Training accuracy >> Validation accuracy

**Solutions**:
1. **Increase dropout rate**:
   ```python
   layers.Dropout(0.7)  # Instead of 0.5
   ```

2. **Add more dropout layers**:
   ```python
   layers.Dropout(0.3)  # After each Conv2D
   ```

3. **Use data augmentation**:
   ```python
   from tensorflow.keras.preprocessing.image import ImageDataGenerator
   ```

4. **Reduce model complexity**:
   ```python
   # Use fewer filters: [16, 32] instead of [32, 64]
   ```

5. **Early stopping**:
   ```python
   python main.py --patience 3
   ```

---

### Problem: Underfitting
**Symptoms**: Both training and validation accuracy are low

**Solutions**:
1. **Increase model capacity**:
   ```python
   # Add more filters: [64, 128]
   # Add more layers
   ```

2. **Train longer**:
   ```python
   python main.py --epochs 20
   ```

3. **Increase learning rate**:
   ```python
   python main.py --learning-rate 0.01
   ```

4. **Remove excessive regularization**:
   ```python
   layers.Dropout(0.2)  # Instead of 0.5
   ```

---

## 💾 Memory Issues

### Problem: Out of memory (OOM)
**Symptoms**:
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions**:
1. **Reduce batch size**:
   ```python
   python main.py --batch-size 16
   ```

2. **Clear session between runs**:
   ```python
   from tensorflow.keras import backend as K
   K.clear_session()
   ```

3. **Enable memory growth** (for GPU):
   ```python
   import tensorflow as tf
   gpus = tf.config.list_physical_devices('GPU')
   for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
   ```

4. **Use gradient accumulation**:
   ```python
   # Train with smaller batches, accumulate gradients
   ```

---

## 📊 Visualization Issues

### Problem: Plots not showing
**Symptoms**: Code runs but no plots appear

**Solutions**:
1. **Check save path**:
   ```python
   import os
   os.makedirs('outputs', exist_ok=True)
   ```

2. **Use plt.show() for interactive**:
   ```python
   import matplotlib.pyplot as plt
   plt.show()
   ```

3. **For Jupyter notebooks**:
   ```python
   %matplotlib inline
   ```

4. **Check file was saved**:
   ```bash
   ls outputs/
   ```

---

### Problem: Feature maps visualization fails
**Symptoms**:
```
ValueError: Layer 'conv1' not found
```

**Solutions**:
1. **Check layer names**:
   ```python
   for layer in model.layers:
       print(layer.name)
   ```

2. **Use correct layer name**:
   ```python
   visualize_feature_maps(model, image, layer_names=['conv1', 'conv2'])
   ```

---

## 🔮 Prediction Issues

### Problem: Poor predictions on custom images
**Symptoms**: Model works on MNIST but fails on your images

**Solutions**:
1. **Ensure correct preprocessing**:
   ```python
   # Image must be:
   # - Grayscale
   # - 28x28 pixels
   # - White digit on black background (like MNIST)
   # - Normalized to [0, 1]
   ```

2. **Invert colors if needed**:
   ```python
   image = 255 - image  # If you have black digit on white
   ```

3. **Center the digit**:
   ```python
   # MNIST digits are centered in 28x28 frame
   ```

4. **Check preprocessing**:
   ```python
   from src.predict import preprocess_image
   processed = preprocess_image(your_image)
   print(processed.shape, processed.min(), processed.max())
   ```

---

## 📁 File and Path Issues

### Problem: Model file not found
**Symptoms**:
```
FileNotFoundError: Model file not found: models/best_model.keras
```

**Solutions**:
1. **Train model first**:
   ```bash
   python main.py
   ```

2. **Check file exists**:
   ```bash
   ls models/
   ```

3. **Use correct path**:
   ```python
   python main.py --model-path models/best_model.keras
   ```

---

### Problem: Import errors from src/
**Symptoms**:
```
ModuleNotFoundError: No module named 'src'
```

**Solutions**:
1. **Run from project root**:
   ```bash
   cd MNIST_CNN_Classification
   python main.py
   ```

2. **Add to Python path**:
   ```python
   import sys
   sys.path.append('.')
   ```

---

## 🐛 Debugging Tips

### General Debugging Strategy

1. **Check shapes at each step**:
   ```python
   print(f"Shape: {x.shape}")
   ```

2. **Verify data ranges**:
   ```python
   print(f"Min: {x.min()}, Max: {x.max()}")
   ```

3. **Test with small data**:
   ```python
   x_small = x_train[:100]
   y_small = y_train[:100]
   ```

4. **Use verbose mode**:
   ```python
   model.fit(..., verbose=2)
   ```

5. **Enable TensorFlow logging**:
   ```python
   import os
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
   ```

---

## 🆘 Getting Help

If you're still stuck:

1. **Check error message carefully**: Read the full traceback
2. **Search the error**: Google the exact error message
3. **Check TensorFlow docs**: [tensorflow.org](https://www.tensorflow.org)
4. **Ask on forums**:
   - Stack Overflow (tag: tensorflow, keras)
   - TensorFlow GitHub issues
   - Reddit r/MachineLearning

5. **Provide context when asking**:
   - Error message (full traceback)
   - Code that produces the error
   - TensorFlow version
   - Operating system

---

## ✅ Prevention Checklist

Before running code:
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Running from project root directory
- [ ] Data shapes are correct
- [ ] Data is normalized
- [ ] Model architecture is valid
- [ ] Output directories exist

---

**Remember**: Most issues are simple fixes! Read error messages carefully and check the basics first. 🔍**
