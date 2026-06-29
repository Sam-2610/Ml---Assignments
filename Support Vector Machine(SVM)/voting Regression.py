from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Generate synthetic regression dataset
x, y = make_regression(
    n_samples=500,
    n_features=20,
    n_informative=5,
    random_state=42
)

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# Initialize individual base regressors
lin_reg = LinearRegression()
dtr = DecisionTreeRegressor(max_depth=3)
svr = SVR()

# Initialize the Voting Regressor
vr = VotingRegressor(
    estimators=[
        ("lr", lin_reg),
        ("dtr", dtr),
        ("svr", svr)
    ]
)

# Fit the model
vr.fit(x_train, y_train)

# Generate predictions
y_pred = vr.predict(x_test)

# Evaluate the model
print("r2 Score : ", r2_score(y_test, y_pred))