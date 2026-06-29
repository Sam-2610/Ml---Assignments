# Machine Learning Models — Ensemble & SVM Collection

A collection of Python scripts covering ensemble learning methods and Support Vector Machines using `scikit-learn` and `xgboost`.

---

## 📁 Files Overview

| File | Algorithm | Task |
|------|-----------|------|
| `svm_classifier.py` | Support Vector Machine | Classification |
| `svm_regressor.py` | Support Vector Regression | Regression |
| `RandomForest.py` | Random Forest | Classification |
| `Bagging_Classifier.py` | Bagging | Classification |
| `Adaptive_Boosting.py` | AdaBoost | Classification |
| `Gradient_Boosting_Regressor.py` | Gradient Boosting | Regression + Classification |
| `XG_Boost.py` | XGBoost | Classification |
| `Voting_Classification.py` | Voting Classifier | Classification |
| `voting_Regression.py` | Voting Regressor | Regression |
| `stacking_classifier.py` | Stacking Classifier + Voting Regressor | Classification + Regression |

---

## 🔧 Requirements

```bash
pip install scikit-learn xgboost pandas numpy
```

---

## 📄 Script Details

### `svm_classifier.py`
Trains an SVM on the **Iris dataset** and benchmarks four kernels — RBF (default), Linear, Polynomial, and Sigmoid. Also sweeps over C values (`0.5` to `5`) with the RBF kernel to observe the effect of regularization on accuracy.

### `svm_regressor.py`
Trains an SVR on the **Diabetes dataset** across all four kernels. Includes a `GridSearchCV` sweep over `C`, `kernel`, and `epsilon`, then re-fits the best configuration. Also demonstrates `LinearSVR` as a faster linear alternative. Target values are scaled with `StandardScaler` before training.

### `RandomForest.py`
Trains a `RandomForestClassifier` with 501 trees and `max_depth=4` on a synthetic classification dataset. Reports both **OOB (Out-of-Bag) score** and test accuracy.

### `Bagging_Classifier.py`
Demonstrates bagging with two different base models — a `DecisionTreeClassifier` and a `LogisticRegression` — using 201 estimators each. Shows how bagging reduces variance regardless of the base learner.

### `Adaptive_Boosting.py`
Uses a **Decision Tree stump** (`max_depth=1`) as the base estimator inside `AdaBoostClassifier` with 100 estimators on a synthetic dataset. Reports accuracy and a full classification report.

### `Gradient_Boosting_Regressor.py`
Two-part script:
- **Part 1 — Regression:** Fits a `GradientBoostingRegressor` (200 trees, `learning_rate=0.05`, `subsample=0.8`) and reports R² score.
- **Part 2 — Classification:** Fits a `GradientBoostingClassifier` (150 trees, `learning_rate=0.1`) on a synthetic classification dataset and reports accuracy.

### `XG_Boost.py`
Trains an `XGBClassifier` with 100 estimators and `max_depth=3` on a synthetic dataset. Uses `logloss` as the eval metric and reports accuracy plus a full classification report.

### `Voting_Classification.py`
Combines `LogisticRegression`, `SVC`, and `DecisionTreeClassifier` into a `VotingClassifier` (hard voting by default). Reports accuracy and classification report.

> ⚠️ **Known bug:** `train_test_split` unpacking order is incorrect — `x_train, y_train, x_test, y_test` should be `x_train, x_test, y_train, y_test`. This will cause a runtime error.

### `voting_Regression.py`
Combines `LinearRegression`, `DecisionTreeRegressor`, and `SVR` into a `VotingRegressor`. Evaluates on a synthetic regression dataset using R² score.

### `stacking_classifier.py`
Two-part script:
- **Classification:** Stacks `LogisticRegression`, `SVC`, and `DecisionTreeClassifier` with a `LogisticRegression` meta-model on the **Iris dataset** using 5-fold CV.
- **Regression:** Uses a `VotingRegressor` with `LinearRegression`, `DecisionTreeRegressor`, and `SVR` on the **Diabetes dataset**.

---

## 🧠 Concepts at a Glance

| Technique | Idea | Reduces |
|-----------|------|---------|
| Bagging | Train models on random data subsets in parallel | Variance |
| Random Forest | Bagging + random feature subsets | Variance |
| AdaBoost | Sequentially up-weight misclassified samples | Bias |
| Gradient Boosting | Sequentially fit residuals | Bias |
| XGBoost | Optimized gradient boosting with regularization | Bias + Variance |
| Voting | Average/majority vote across models | Both |
| Stacking | Use model predictions as features for a meta-model | Both |
| SVM/SVR | Find the maximum-margin hyperplane | — |

---

## ▶️ Usage

Each script is self-contained and generates its own synthetic (or built-in) dataset. Run any file directly:

```bash
python Adaptive_Boosting.py
python XG_Boost.py
python svm_classifier.py
# etc.
```