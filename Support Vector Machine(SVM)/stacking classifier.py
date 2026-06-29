from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_diabetes
from sklearn.metrics import r2_score, accuracy_score

# ===== Classification Example =====
# Load classification data
iris = load_iris()
x_class = iris.data
y_class = iris.target

# Define base classifiers
lr = LogisticRegression(max_iter=200, random_state=42)
svc = SVC(kernel='rbf', random_state=42)
dtc = DecisionTreeClassifier(random_state=42)

# Define meta model
meta_model = LogisticRegression(random_state=42)

# Create stacking classifier
stacking_clf = StackingClassifier(
    estimators=[
        ("lr", lr),
        ("svc", svc),
        ("dtc", dtc)
    ],
    final_estimator=meta_model,
    cv=5
)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x_class, y_class, test_size=0.3, random_state=42
)

# Fit and predict
stacking_clf.fit(x_train, y_train)
y_pred = stacking_clf.predict(x_test)

print("Classification - Accuracy Score : ", accuracy_score(y_test, y_pred))

# ===== Regression Example =====
# Load regression data
diabetes = load_diabetes()
x_reg = diabetes.data
y_reg = diabetes.target

# Define base regressors
lin_reg = LinearRegression()
dtr = DecisionTreeRegressor(random_state=42)
svr = SVR(kernel='rbf')

# Create voting regressor
vr = VotingRegressor(
    estimators=[
        ("lr", lin_reg),
        ("dtr", dtr),
        ("svr", svr)
    ]
)

# Split data for regression
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
    x_reg, y_reg, test_size=0.3, random_state=42
)

# Fit and predict
vr.fit(x_train_reg, y_train_reg)
y_pred_train_reg = vr.predict(x_train_reg)
y_pred_test_reg = vr.predict(x_test_reg)

print("Regression - R2 Score (Test) : ", r2_score(y_test_reg, y_pred_test_reg))
print("Regression - R2 Score (Train) : ", r2_score(y_train_reg, y_pred_train_reg))