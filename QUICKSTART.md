# 🚀 QUICKSTART GUIDE

## Run in 3 Steps

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Project
```bash
python3 main.py
```

### 3️⃣ Check Results
Look in the `outputs/` folder for visualizations!

---

## What Happens When You Run?

```
STEP 1: Load MNIST data (60,000 training images)
STEP 2: Create visualizations of the data
STEP 3: Build neural network model
STEP 4: Train for 5 epochs (~1 minute)
STEP 5: Evaluate on test data
STEP 6: Generate result visualizations
```

---

## Expected Output

```
Test Accuracy: ~89-90%
Training Time: ~1 minute
```

---

## Files Generated

**In `models/` folder:**
- `mnist_model.keras` - Your trained model

**In `outputs/` folder:**
- `sample_images.png` - Example digits
- `class_distribution.png` - Data distribution
- `training_history.png` - Training progress
- `confusion_matrix.png` - Prediction accuracy
- `sample_predictions.png` - Test results

---

## Custom Training

```bash
# Train for 10 epochs
python3 main.py --epochs 10

# Use different batch size
python3 main.py --batch-size 64

# Adjust learning rate
python3 main.py --learning-rate 0.01
```

---

## Troubleshooting

**Problem**: `No module named 'tensorflow'`  
**Solution**: Run `pip install -r requirements.txt`

**Problem**: Training too slow  
**Solution**: Use fewer epochs: `python3 main.py --epochs 3`

---

That's it! Simple and straightforward. 🎉
