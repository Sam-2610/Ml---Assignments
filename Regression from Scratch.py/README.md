# ML Algorithms from Scratch

Three core machine learning algorithms implemented from scratch using only NumPy, without any scikit-learn model classes. Built to understand the math and mechanics behind each algorithm rather than just calling `.fit()` on a library.

---

## Files Overview

### 1. `Linear_Regression.py`
Implements Linear Regression using **gradient descent** from scratch.

**How it works:**
- Forward pass: `y_pred = bias + X · weights`
- Loss: Mean Squared Error (MSE)
- Gradients: `dw = (1/m) * Xᵀ · (y_pred - y)` and `db = (1/m) * Σ(y_pred - y)`
- Parameters updated each iteration: `w -= lr * dw`, `b -= lr * db`

**Test case:** Learns `y = 2x` perfectly from 5 data points in 1000 iterations.

```
Input  : [[1], [2], [3], [4], [5]]
Target : [2, 4, 6, 8, 10]
Output : [ 2.  4.  6.  8. 10.]  ← converges exactly
```

---

### 2. `Logistic_Regression.py`
Implements Logistic Regression for **binary classification** using gradient descent and the sigmoid function.

**How it works:**
- Forward pass: `z = bias + X · weights`
- Activation: `sigmoid(z) = 1 / (1 + e⁻ᶻ)` → outputs probability between 0 and 1
- Loss: Binary Cross-Entropy (gradients derived from it)
- Prediction: probability `>= 0.5` → class 1, else class 0

**Key methods:**
| Method | Purpose |
|---|---|
| `fit(x, y)` | Trains weights and bias via gradient descent |
| `get_probabilities(x)` | Returns raw sigmoid probabilities |
| `predict(x, threshold)` | Returns class labels (0 or 1) |

**Test case:** Separates `[0, 0, 1, 1]` classes from 2-feature input.

---

### 3. `Kn_Neighbours.py`
Implements K-Nearest Neighbors classifier from scratch using **Euclidean distance** and majority voting.

**How it works:**
- For each test point, compute distance to every training point
- Pick the `k` closest training points
- Return the majority class among those `k` neighbours

**Key methods:**
| Method | Purpose |
|---|---|
| `fit(x, y)` | Stores training data (KNN has no training step) |
| `__euclidean_dist(x1, x2)` | Computes `√Σ(x1 - x2)²` |
| `__predict_one(x)` | Predicts class for a single point |
| `predict(x)` | Predicts class for all test points |

**Test case:** 5 training points (2 classes), 2 test points — also visualizes training vs test points using Seaborn.

```
x_test = [[2,2], [6,6]]
y_pred = [0, 1]   ← correctly classifies both
```

---

## Key Concepts Demonstrated

| Concept | Linear Reg | Logistic Reg | KNN |
|---|---|---|---|
| Gradient Descent | ✅ | ✅ | ❌ (not needed) |
| Sigmoid Activation | ❌ | ✅ | ❌ |
| Euclidean Distance | ❌ | ❌ | ✅ |
| Majority Voting | ❌ | ❌ | ✅ |
| Bias + Weights | ✅ | ✅ | ❌ |
| Visualization | ❌ | ❌ | ✅ |

---

## Dependencies

```bash
pip install numpy pandas seaborn matplotlib
```

> `Linear_Regression.py` and `Logistic_Regression.py` only require **NumPy**.
> `Kn_Neighbours.py` also uses Pandas, Seaborn, and Matplotlib for visualization.

---

## Run

```bash
python Linear_Regression.py
python Logistic_Regression.py
python Kn_Neighbours.py
```

All three files include an `if __name__ == "__main__"` block with test data, so they can be run directly.

---

## Why from Scratch?

Using scikit-learn is the right choice for real projects — but implementing algorithms from scratch builds a much deeper understanding of:
- Why scaling matters for gradient-based models
- What the sigmoid function actually does to a linear output
- Why KNN gets slow on large datasets (it computes distances to every training point)
- Where the gradient formulas come from

---

## Author

**Satyam Sagar**
satyamsagar@gmail.com