# Machine Learning Practice — Regression & Classification

A collection of six focused practice files covering core supervised learning algorithms in scikit-learn, applied to real-world datasets (heart disease prediction and insurance charges estimation).

---

## Files Overview

### 1. `Linear_Regression.py`
Predicts insurance charges from patient demographics using Linear Regression.

| Topic | Detail |
|---|---|
| Dataset | `insurance.csv` |
| Target | `charges` (continuous) |
| Encoding | Label encoding for `sex`, `smoker`; One-Hot for `region` |
| Feature Engineering | `age_smoker`, `bmi_smoker` interaction terms |
| Evaluation | R², Adjusted R², overfitting check |

- Visualizes BMI vs Charges (colored by smoker status)
- Computes both training and test R² to detect overfitting/underfitting

---

### 2. `Lasso_Regression.py`
Extends linear regression with L1 regularization to reduce overfitting and perform feature selection.

| Topic | Detail |
|---|---|
| Dataset | `insurance.csv` |
| Target | `charges` (continuous) |
| Models | `Lasso` (manual alpha search) + `LassoCV` (auto tuning) |
| Evaluation | MSE, R² |

- Manually tests alphas `[0.001, 0.1, 1, 2, 5, 10, 25]` and plots Alpha vs MSE
- Uses `LassoCV` with 5-fold CV to automatically find the best alpha

---

### 3. `Logistic_Regression.py`
Predicts heart disease (binary classification) using Logistic Regression, with a KNN comparison using GridSearchCV.

| Topic | Detail |
|---|---|
| Dataset | `heart.csv` |
| Target | `target` (0 or 1) |
| Models | Logistic Regression (unscaled + scaled) + KNN with GridSearchCV |
| Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report |

- Compares unscaled vs scaled Logistic Regression performance
- Tunes KNN with `n_neighbors ∈ [3, 5, 7, 9]` using recall as the scoring metric

---

### 4. `KNN_Nearest_Neighbours.py`
Dedicated K-Nearest Neighbors practice with manual k selection and GridSearchCV tuning.

| Topic | Detail |
|---|---|
| Dataset | `heart.csv` |
| Target | `target` (0 or 1) |
| Models | KNN (k=7 manual) + KNN with GridSearchCV |
| Evaluation | Accuracy, Precision, Recall |

- Manually tests `k=7`, then uses GridSearchCV to find the best k
- Prints CV results table showing recall score per k value

---

### 5. `Naive_Bayes.py`
Predicts heart disease using Gaussian Naive Bayes — the simplest, fastest baseline classifier.

| Topic | Detail |
|---|---|
| Dataset | `heart.csv` |
| Target | `target` (0 or 1) |
| Model | `GaussianNB` |
| Evaluation | Accuracy, Recall, Precision |

- No scaling required (GaussianNB is scale-invariant)
- Useful as a probabilistic baseline to compare against KNN and Logistic Regression

---

### 6. `Pipeline_in_Sklearn.py`
Demonstrates the correct way to combine preprocessing and modeling using scikit-learn `Pipeline` with `GridSearchCV`.

| Topic | Detail |
|---|---|
| Dataset | `heart.csv` |
| Target | `target` (0 or 1) |
| Pipeline Steps | `StandardScaler` → `KNeighborsClassifier` |
| Evaluation | Accuracy, Recall, Precision |

- **Prevents data leakage** — scaler fits only on training folds during cross-validation
- Uses `knn__n_neighbors` double-underscore syntax to pass params into pipeline steps
- Recommended pattern for production-quality ML pipelines

---

## Datasets Required

Place these CSV files in the same folder as the scripts before running:

| File | Used In |
|---|---|
| `heart.csv` | Logistic Regression, KNN, Naive Bayes, Pipeline |
| `insurance.csv` | Linear Regression, Lasso Regression |

---

## Skills Practiced

- Supervised learning: regression and classification
- Data encoding: label encoding, one-hot encoding
- Feature engineering: interaction terms
- Model evaluation: R², MSE, Accuracy, Precision, Recall, F1, Confusion Matrix
- Regularization with Lasso (L1)
- Hyperparameter tuning with `GridSearchCV`
- Cross-validation
- Preventing data leakage using `Pipeline`

---

## Setup

```bash
pip install pandas scikit-learn seaborn matplotlib
```

Run any file individually:

```bash
python Linear_Regression.py
python Lasso_Regression.py
python Logistic_Regression.py
python KNN_Nearest_Neighbours.py
python Naive_Bayes.py
python Pipeline_in_Sklearn.py
```

---

## Author

**Satyam Sagar**
satyamsagar827@gmail.com