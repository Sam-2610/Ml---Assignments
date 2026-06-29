# ─────────────────────────────────────────────
# Import Modules
# ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


# ═════════════════════════════════════════════
# PART 1 — Gradient Boosting Regression
# ═════════════════════════════════════════════

# ─────────────────────────────────────────────
# 1. Generate Regression Data
# ─────────────────────────────────────────────
# make_regression creates a synthetic regression dataset
# n_features=10 : 10 input features
# noise=20      : adds random noise to make it realistic
x, y = make_regression( # type: ignore
    n_samples=1000,
    n_features=10,
    noise=20,
    random_state=42
)

# ─────────────────────────────────────────────
# 2. Train Test Split
# ─────────────────────────────────────────────
# Bug fixed — correct order is x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 3. Train Gradient Boosting Regressor
# ─────────────────────────────────────────────
# n_estimators  : number of boosting stages (trees)
# learning_rate : shrinks each tree's contribution (lower = slower but better)
# max_depth     : depth of each individual tree (shallow trees work best in boosting)
# subsample     : fraction of samples used per tree (0.8 = 80%, adds randomness)
gbr = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
gbr.fit(x_train, y_train)

# ─────────────────────────────────────────────
# 4. Evaluate Regressor
# ─────────────────────────────────────────────
y_pred = gbr.predict(x_test)
print("R² Score (Regression) : ", r2_score(y_test, y_pred))


# ═════════════════════════════════════════════
# PART 2 — Gradient Boosting Classification
# ═════════════════════════════════════════════

# ─────────────────────────────────────────────
# 5. Generate Classification Data
# ─────────────────────────────────────────────
# make_classification creates a synthetic binary classification dataset
# n_informative=10 : only 10 of the 20 features are actually useful
x, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=10,
    random_state=42
)

# ─────────────────────────────────────────────
# 6. Train Test Split
# ─────────────────────────────────────────────
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# ─────────────────────────────────────────────
# 7. Train Gradient Boosting Classifier
# ─────────────────────────────────────────────
# Same boosting concept as regression but optimizes log-loss for classification
# Higher learning_rate here since dataset is smaller (fewer samples)
gbc = GradientBoostingClassifier(
    learning_rate=0.1,
    n_estimators=150,
    max_depth=3,
    random_state=42
)
gbc.fit(x_train, y_train)

# ─────────────────────────────────────────────
# 8. Evaluate Classifier
# ─────────────────────────────────────────────
y_pred = gbc.predict(x_test)
print("Accuracy (Classification) : ", accuracy_score(y_test, y_pred))