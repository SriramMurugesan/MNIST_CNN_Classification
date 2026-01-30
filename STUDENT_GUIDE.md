# 🎓 MNIST CNN Classification - Student Instructions

## 📋 Before You Start

This project will teach you how to build a Convolutional Neural Network (CNN) to recognize handwritten digits. You'll learn CNN fundamentals, training techniques, and model evaluation.

---

## ⚙️ Setup (First Time Only)

### Step 1: Verify Python Installation
```bash
python3 --version
```
You need Python 3.8 or higher.

### Step 2: Navigate to Project Directory
```bash
cd MNIST_CNN_Classification
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
This installs TensorFlow, NumPy, Matplotlib, and other required packages.

### Step 4: Verify Installation
```bash
python3 verify_installation.py
```
This checks that everything is installed correctly.

---

## 🚀 Quick Start (Run the Complete Project)

### Train the Model
```bash
python3 main.py
```

This will:
1. Load and preprocess MNIST data (60,000 training images)
2. Create a CNN model
3. Train for 10 epochs (~5-10 minutes)
4. Evaluate on test set
5. Generate visualizations in `outputs/` folder
6. Save trained model in `models/` folder

### Expected Output
```
Training Accuracy: ~99%
Test Accuracy: ~98-99%
```

---

## 📚 Learning Path

### Option 1: Interactive Learning (Recommended for Beginners)

Start with Jupyter notebooks for step-by-step learning:

```bash
jupyter notebook
```

Then open notebooks in this order:
1. `01_introduction.ipynb` - Understand the MNIST dataset
2. `02_preprocessing.ipynb` - Learn data preprocessing (create this)
3. `03_cnn_architecture.ipynb` - Understand CNN layers (create this)
4. `04_training.ipynb` - Learn training process (create this)
5. `05_evaluation.ipynb` - Analyze model performance (create this)

### Option 2: Code Exploration (For Intermediate Students)

Read the source code in this order:

1. **`src/data_loader.py`** - How data is loaded and preprocessed
2. **`src/model.py`** - CNN architecture definition
3. **`src/train.py`** - Training pipeline
4. **`src/evaluate.py`** - Model evaluation
5. **`src/predict.py`** - Making predictions
6. **`src/visualize.py`** - Visualization utilities

Each file has detailed comments explaining every step.

### Option 3: Experimentation (For Advanced Students)

Run experiments with different parameters:

```bash
# Try different learning rates
python3 main.py --learning-rate 0.0001
python3 main.py --learning-rate 0.01

# Try different batch sizes
python3 main.py --batch-size 16
python3 main.py --batch-size 128

# Train for more epochs
python3 main.py --epochs 20

# Use different optimizer
python3 main.py --optimizer sgd
```

---

## 📊 Understanding the Outputs

After running `python3 main.py`, check these files:

### Models (in `models/` folder)
- `best_model.keras` - Best performing model during training
- `final_model.keras` - Final model after all epochs

### Visualizations (in `outputs/` folder)
- `training_history.png` - Shows how accuracy improved over time
- `confusion_matrix.png` - Shows which digits are confused
- `misclassified_samples.png` - Examples of wrong predictions
- `sample_images.png` - Random samples from dataset
- `class_distribution.png` - How many samples per digit
- `feature_maps_*.png` - What the CNN "sees"
- `filters.png` - What patterns the CNN learned

---

## 🎯 Exercises to Complete

See `docs/exercises.md` for detailed assignments.

### Week 1: Basics
- Exercise 1: Data Exploration
- Exercise 2: Preprocessing Impact
- Exercise 3: Model Architecture Analysis

### Week 2: Intermediate
- Exercise 4: Hyperparameter Tuning
- Exercise 5: Architecture Modification
- Exercise 6: Dropout Experimentation

### Week 3: Advanced
- Exercise 7: Data Augmentation
- Exercise 8: Transfer Learning
- Exercise 9: Error Analysis

### Week 4: Challenges
- Challenge 1: Ensemble Model
- Challenge 2: Visualization Dashboard
- Challenge 3: Real-Time Recognition

---

## 🔧 Common Commands

### Training
```bash
# Default training (10 epochs)
python3 main.py

# Quick test (3 epochs)
python3 main.py --epochs 3

# Longer training (20 epochs)
python3 main.py --epochs 20

# Custom parameters
python3 main.py --epochs 15 --batch-size 64 --learning-rate 0.001
```

### Using Existing Model
```bash
# Skip training, use saved model
python3 main.py --no-train

# Just create visualizations
python3 main.py --visualize-only
```

### Testing Individual Modules
```bash
# Test data loader
python3 -c "from src.data_loader import load_and_preprocess_data; load_and_preprocess_data()"

# Test model creation
python3 -c "from src.model import create_and_compile_model; create_and_compile_model()"
```

---

## ❓ Troubleshooting

### Problem: "Command 'python' not found"
**Solution:** Use `python3` instead of `python`

### Problem: "No module named 'tensorflow'"
**Solution:** 
```bash
pip install -r requirements.txt
```

### Problem: Training is very slow
**Solution:** 
```bash
# Use smaller batch size or fewer epochs
python3 main.py --epochs 3 --batch-size 16
```

### Problem: "Out of memory" error
**Solution:**
```bash
# Reduce batch size
python3 main.py --batch-size 16
```

### More Help
See `docs/troubleshooting.md` for detailed solutions.

---

## 📖 Documentation

- **`README.md`** - Complete project overview
- **`QUICKSTART.md`** - Quick reference guide
- **`docs/learning_outcomes.md`** - What you'll learn
- **`docs/exercises.md`** - Practice assignments
- **`docs/troubleshooting.md`** - Problem solving

---

## 🎓 What You'll Learn

By completing this project, you will understand:

1. **CNN Architecture**
   - Convolutional layers and filters
   - Pooling operations
   - Dense layers and dropout

2. **Training Process**
   - Forward and backward propagation
   - Loss functions and optimizers
   - Learning rate and batch size

3. **Model Evaluation**
   - Accuracy and loss metrics
   - Confusion matrix
   - Error analysis

4. **Best Practices**
   - Data preprocessing
   - Preventing overfitting
   - Model checkpointing
   - Visualization

5. **TensorFlow/Keras**
   - Building models
   - Training pipelines
   - Saving and loading models

---

## 🏆 Success Criteria

You've mastered the material when you can:

- [ ] Explain how CNNs work to someone else
- [ ] Achieve >95% test accuracy
- [ ] Interpret the confusion matrix
- [ ] Modify the architecture successfully
- [ ] Debug common training issues
- [ ] Apply the pipeline to a new dataset

---

## 📝 Submission (For Graded Assignments)

Submit the following:

1. **Code**: Your modified scripts or notebooks
2. **Results**: Training curves, confusion matrix, accuracy
3. **Analysis**: 1-2 page report explaining:
   - What you tried
   - What worked/didn't work
   - What you learned
4. **Reflection**: Challenges faced and how you solved them

---

## 🚀 Next Steps After Completion

1. **Try Fashion-MNIST**: Similar dataset with clothing items
2. **Explore CIFAR-10**: Color images, more complex
3. **Build a Web App**: Deploy your model with Streamlit
4. **Kaggle Competitions**: Apply skills to real challenges
5. **Advanced Architectures**: ResNet, VGG, Inception

---

## 💡 Tips for Success

1. **Read the code comments** - They explain the "why" not just "what"
2. **Experiment freely** - Try changing parameters and see what happens
3. **Visualize everything** - Use the visualization tools provided
4. **Ask questions** - If confused, check troubleshooting guide
5. **Take notes** - Document what you learn
6. **Be patient** - Deep learning takes time to understand

---

## 🤝 Getting Help

1. **Check documentation** first (README, troubleshooting guide)
2. **Run verification script**: `python3 verify_installation.py`
3. **Read error messages** carefully - they often tell you the problem
4. **Search online** - Many others have faced similar issues
5. **Ask instructor** or classmates

---

## ✅ Checklist Before Starting

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Verification script passed (`python3 verify_installation.py`)
- [ ] Read README.md
- [ ] Understand the learning objectives

---

**You're ready to start! Good luck and enjoy learning about CNNs! 🎉**

Run this command to begin:
```bash
python3 main.py
```
