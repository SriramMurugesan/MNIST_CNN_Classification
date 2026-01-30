# Exercises and Assignments

## 🎯 Purpose

These exercises are designed to reinforce your understanding of CNNs and the MNIST classification pipeline. Start with basic exercises and progress to advanced challenges.

---

## 📝 Basic Exercises (Beginner Level)

### Exercise 1: Data Exploration
**Objective**: Understand the MNIST dataset structure

**Tasks**:
1. Load the MNIST dataset and print the shapes of training and test sets
2. Display 10 random images from each digit class (0-9)
3. Calculate and plot the distribution of classes in the training set
4. Find the minimum and maximum pixel values in the dataset

**Expected Output**: Visualizations showing sample images and class distribution

---

### Exercise 2: Preprocessing Impact
**Objective**: Understand the importance of normalization

**Tasks**:
1. Train two models: one with normalized data (0-1) and one without (0-255)
2. Compare training time and final accuracy
3. Plot training curves for both models side by side
4. Explain why normalization helps

**Questions to Answer**:
- Which model converges faster?
- What is the accuracy difference?
- Why does normalization matter?

---

### Exercise 3: Model Architecture Analysis
**Objective**: Understand CNN components

**Tasks**:
1. Print the model summary and count total parameters
2. Calculate parameters manually for the first Conv2D layer
3. Explain what each layer does in your own words
4. Draw a diagram of the data flow through the network

**Deliverable**: Document explaining each layer's role and parameter calculations

---

## 🔧 Intermediate Exercises

### Exercise 4: Hyperparameter Tuning
**Objective**: Learn how hyperparameters affect performance

**Tasks**:
1. Train models with different batch sizes: [16, 32, 64, 128]
2. Train models with different learning rates: [0.0001, 0.001, 0.01]
3. Compare training time and final accuracy for each configuration
4. Create a table summarizing results

**Analysis**:
- Which batch size works best? Why?
- Which learning rate is optimal?
- What trade-offs did you observe?

---

### Exercise 5: Architecture Modification
**Objective**: Experiment with different architectures

**Tasks**:
1. **Shallow Network**: Remove one convolutional block, retrain, compare accuracy
2. **Deeper Network**: Add one more convolutional block, retrain, compare accuracy
3. **Wider Network**: Increase filters to [64, 128] instead of [32, 64]
4. **Different Pooling**: Try AveragePooling2D instead of MaxPooling2D

**Deliverable**: Report comparing all architectures with accuracy, parameters, and training time

---

### Exercise 6: Dropout Experimentation
**Objective**: Understand regularization

**Tasks**:
1. Train models with dropout rates: [0.0, 0.3, 0.5, 0.7]
2. Plot training vs validation accuracy for each
3. Identify which rate prevents overfitting best
4. Explain the relationship between dropout rate and overfitting

**Expected Insight**: Understanding the bias-variance tradeoff

---

## 🚀 Advanced Exercises

### Exercise 7: Data Augmentation
**Objective**: Improve model generalization

**Tasks**:
1. Implement data augmentation (rotation, shift, zoom)
2. Train model with and without augmentation
3. Compare test accuracy and overfitting behavior
4. Visualize augmented samples

**Code Hint**:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
```

---

### Exercise 8: Transfer Learning Simulation
**Objective**: Understand feature reuse

**Tasks**:
1. Train a model on digits 0-4 only
2. Freeze convolutional layers
3. Retrain only dense layers on digits 5-9
4. Compare with training from scratch on digits 5-9

**Analysis**: Does pre-training help? Why or why not?

---

### Exercise 9: Error Analysis Deep Dive
**Objective**: Understand model failures

**Tasks**:
1. Find the 20 most confident wrong predictions
2. Find the 20 least confident correct predictions
3. Analyze patterns: which digits are confused most?
4. Propose architecture changes to fix specific errors

**Deliverable**: Detailed error analysis report with visualizations

---

### Exercise 10: Custom Dataset Application
**Objective**: Apply knowledge to new problem

**Tasks**:
1. Collect or download a different digit dataset (e.g., handwritten digits you create)
2. Preprocess to match MNIST format
3. Test the trained MNIST model on your dataset
4. Fine-tune the model on your dataset

**Challenge**: Achieve >90% accuracy on your custom dataset

---

## 🏆 Challenge Projects

### Challenge 1: Ensemble Model
**Objective**: Combine multiple models

**Tasks**:
1. Train 5 models with different random initializations
2. Implement voting ensemble (majority vote)
3. Implement averaging ensemble (average probabilities)
4. Compare ensemble accuracy with single model

**Goal**: Beat single model accuracy by at least 0.5%

---

### Challenge 2: Visualization Dashboard
**Objective**: Create interactive visualization tool

**Tasks**:
1. Build a function that takes an image and shows:
   - Original image
   - Feature maps from each layer
   - Prediction probabilities
   - Top-3 predictions
2. Make it work for any input image
3. Create a grid showing this for 10 test images

**Bonus**: Use Plotly for interactive visualizations

---

### Challenge 3: Real-Time Digit Recognition
**Objective**: Deploy model for practical use

**Tasks**:
1. Create a simple web interface (using Streamlit or Gradio)
2. Allow users to draw digits
3. Predict in real-time
4. Display confidence scores

**Deliverable**: Working web application

---

### Challenge 4: Optimize for Speed
**Objective**: Model compression and optimization

**Tasks**:
1. Implement model pruning (remove small weights)
2. Implement quantization (reduce precision)
3. Measure inference time before and after
4. Compare accuracy vs speed tradeoff

**Goal**: Reduce model size by 50% while maintaining >97% accuracy

---

### Challenge 5: Adversarial Examples
**Objective**: Understand model vulnerabilities

**Tasks**:
1. Implement Fast Gradient Sign Method (FGSM)
2. Generate adversarial examples that fool the model
3. Visualize the perturbations
4. Implement adversarial training to defend

**Learning**: Understanding model robustness

---

## 📊 Assignment Rubric

### Grading Criteria

| Criteria | Points | Description |
|----------|--------|-------------|
| **Code Quality** | 25 | Clean, well-documented, modular code |
| **Correctness** | 25 | Accurate implementation and results |
| **Analysis** | 25 | Thoughtful interpretation of results |
| **Visualization** | 15 | Clear, informative plots and figures |
| **Creativity** | 10 | Novel insights or approaches |

---

## 💡 Tips for Success

1. **Start Simple**: Begin with basic exercises before advanced ones
2. **Document Everything**: Keep notes on what you try and results
3. **Visualize Often**: Plots help understand what's happening
4. **Ask Questions**: Why does this work? What if I change X?
5. **Experiment**: Don't be afraid to try new ideas
6. **Compare**: Always have a baseline to compare against
7. **Reproduce**: Make sure your results are reproducible

---

## 🔍 Self-Assessment Questions

After completing exercises, ask yourself:

- [ ] Can I explain how CNNs work to someone else?
- [ ] Do I understand why each layer is necessary?
- [ ] Can I debug common training issues?
- [ ] Can I modify the architecture for different tasks?
- [ ] Do I know how to prevent overfitting?
- [ ] Can I interpret confusion matrices and error patterns?
- [ ] Am I comfortable with TensorFlow/Keras?

---

## 📚 Additional Resources

### For Deeper Understanding
- **3Blue1Brown**: Neural Networks series (YouTube)
- **CS231n**: Stanford's CNN course notes
- **Distill.pub**: Interactive ML explanations
- **Papers with Code**: State-of-the-art implementations

### Practice Datasets
- Fashion-MNIST (clothing classification)
- CIFAR-10 (color images)
- SVHN (street view house numbers)
- Kaggle digit recognizer competition

---

## 🎓 Submission Guidelines

For each exercise, submit:
1. **Code**: Well-commented Python scripts or notebooks
2. **Results**: Accuracy metrics, training curves, visualizations
3. **Analysis**: Written explanation of findings (1-2 paragraphs)
4. **Reflection**: What you learned and challenges faced

**Format**: Jupyter Notebook (.ipynb) or Python script (.py) + PDF report

---

**Good luck with your exercises! Remember: The goal is learning, not just completing tasks. Take time to understand each concept deeply. 🚀**
