# Decision Tree — Classification & Regression

Two practice files exploring Decision Tree models in scikit-learn, covering the full workflow from data preparation to pruning strategies. Applied to real-world datasets: Titanic survival prediction (classification) and diabetes progression prediction (regression).

---

## Files Overview

### 1. `Decision_Tree_Classifier.py`
Predicts Titanic passenger survival using a Decision Tree Classifier, with three pruning strategies compared.

| Topic | Detail |
|---|---|
| Dataset | `titanic` (loaded via Seaborn) |
| Target | `survived` (0 = Died, 1 = Survived) |
| Features | `pclass`, `sex`, `fare`, `embarked`, `age` |
| Evaluation | Accuracy Score |

**Workflow:**

- **Missing values** — `age` imputed with median, `embarked` with most frequent value
- **Encoding** — `sex` and `embarked` label-encoded to numeric values
- **No Pruning** — full unpruned tree trained and visualized (will overfit)
- **Pre Pruning (max_depth)** — tests depths `[2–10]`, prints accuracy per depth
- **Pre Pruning (min_samples_split)** — tests split values `[10, 15, 20, 25, 30]` with `max_depth=4`
- **Post Pruning (ccp_alpha)** — extracts effective alphas from the pruning path, trains one tree per alpha, selects the best

---

### 2. `Decision_Tree_Regressor.py`
Predicts diabetes disease progression (a continuous value) using a Decision Tree Regressor.

| Topic | Detail |
|---|---|
| Dataset | `diabetes` (loaded via scikit-learn) |
| Target | Disease progression score (continuous) |
| Features | Age, BMI, blood pressure, and 7 other clinical measurements |
| Evaluation | MSE, R² (train and test) |

**Workflow:**

- Loads the diabetes dataset directly as a pandas DataFrame
- Trains a full unpruned `DecisionTreeRegressor`
- Evaluates on both train and test sets to expose overfitting
- Visualizes the tree structure with `plot_tree`

**Expected output pattern (unpruned tree):**

| Metric | Train | Test |
|---|---|---|
| MSE | ≈ 0.0 | High (~5000+) |
| R² | ≈ 1.0 | Low (~0.1–0.3) |

This demonstrates why pruning is necessary — the tree memorizes training data perfectly but generalizes poorly.

---

## Pruning Strategies Explained

| Strategy | How | When to Use |
|---|---|---|
| **No Pruning** | Default tree, grows until pure leaves | Baseline only |
| **Pre Pruning — max_depth** | Limits how deep the tree can grow | Quick control over complexity |
| **Pre Pruning — min_samples_split** | Minimum samples needed to split a node | Prevents splits on very small groups |
| **Post Pruning — ccp_alpha** | Removes branches after full tree is built | More precise, data-driven pruning |

---

## Dataset Details

| Dataset | Source | Task | Records |
|---|---|---|---|
| Titanic | `sns.load_dataset("titanic")` | Binary Classification | 891 |
| Diabetes | `sklearn.datasets.load_diabetes` | Regression | 442 |

No external CSV files needed — both datasets load automatically.

---

## Skills Practiced

- Decision Tree classification and regression
- Handling missing values with `SimpleImputer`
- Label encoding with `LabelEncoder`
- Tree visualization with `plot_tree`
- Pre-pruning with `max_depth` and `min_samples_split`
- Post-pruning with `cost_complexity_pruning_path` and `ccp_alpha`
- Detecting overfitting by comparing train vs test metrics
- Model evaluation: Accuracy, MSE, R²

---

## Setup

```bash
pip install scikit-learn seaborn matplotlib pandas
```

```bash
python Decision_Tree_Classifier.py
python Decision_Tree_Regressor.py
```

---

## Author

**Satyam Sagar**
satyamsagar@gmail.com