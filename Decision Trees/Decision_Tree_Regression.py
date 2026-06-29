# ─────────────────────────────────────────────
# Import Modules
# ─────────────────────────────────────────────
import matplotlib.pyplot as plt                          # Bug 1 fixed — removed duplicate import
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error

# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
# load_diabetes returns a regression dataset with 10 features
# as_frame=True gives us a pandas DataFrame directly
df = load_diabetes(as_frame=True).frame # type: ignore

x = df.drop("target", axis=1)   # Features: age, bmi, blood pressure, etc.
y = df["target"]                 # Target: disease progression score (continuous)

# ─────────────────────────────────────────────
# 2. Train Test Split
# ─────────────────────────────────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# ─────────────────────────────────────────────
# 3. Decision Tree Regressor — No Pruning
# ─────────────────────────────────────────────
# Default DecisionTreeRegressor grows a full unpruned tree
# This often leads to overfitting (perfect train score, poor test score)
model = DecisionTreeRegressor(random_state=42)
model.fit(x_train, y_train)

# ─────────────────────────────────────────────
# 4. Predictions
# ─────────────────────────────────────────────
y_pred_train = model.predict(x_train)
y_pred_test  = model.predict(x_test)

# ─────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────
# MSE — lower is better; expect train MSE ≈ 0 (overfitting)
print("MSE Train : ", mean_squared_error(y_train, y_pred_train))
print("MSE Test  : ", mean_squared_error(y_test, y_pred_test))

# R² — closer to 1 is better; train will be 1.0, test will be much lower
print("R² Train  : ", r2_score(y_train, y_pred_train))
print("R² Test   : ", r2_score(y_test, y_pred_test))   # Bug 2 fixed — was using y_train

# ─────────────────────────────────────────────
# 6. Plot the Tree
# ─────────────────────────────────────────────
plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=x.columns,
    # Bug 3 fixed — removed class_names (only valid for classifiers, not regressors)
    filled=True
)
plt.title("Decision Tree — Regressor (No Pruning)")
plt.tight_layout()
plt.show()