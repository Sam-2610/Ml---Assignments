from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR

df = datasets.load_diabetes(as_frame=True).frame # type: ignore

x = df.drop("target",axis=1)
y = df["target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Scale target values using the training set only

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

# Model

model = SVR()
model.fit(x_train,y_train_scaled)

y_test_pred_scaled = model.predict(x_test)
y_train_pred_scaled = model.predict(x_train)

print(r2_score(y_train_scaled,y_train_pred_scaled))
print(r2_score(y_test_scaled,y_test_pred_scaled))

# Linear Kernel

model = SVR(kernel="linear")
model.fit(x_train,y_train_scaled)

y_test_pred_scaled = model.predict(x_test)
y_train_pred_scaled = model.predict(x_train)

print(r2_score(y_train_scaled,y_train_pred_scaled))
print(r2_score(y_test_scaled,y_test_pred_scaled))

# Polynomial

model = SVR(kernel="poly")
model.fit(x_train,y_train_scaled)

y_test_pred_scaled = model.predict(x_test)
y_train_pred_scaled = model.predict(x_train)

print(r2_score(y_train_scaled,y_train_pred_scaled))
print(r2_score(y_test_scaled,y_test_pred_scaled))

# Sigmoid

model = SVR(kernel="sigmoid")
model.fit(x_train,y_train_scaled)

y_test_pred_scaled = model.predict(x_test)
y_train_pred_scaled = model.predict(x_train)

print(r2_score(y_train_scaled,y_train_pred_scaled))
print(r2_score(y_test_scaled,y_test_pred_scaled))

# GridSearchCV

param_grid = {
    "C": [1, 2, 5, 10, 50, 100],
    "kernel": ["rbf", "linear"],
    "epsilon": [0.01, 0.1, 0.2, 0.3, 0.5]
}

svr = SVR()

grid_search = GridSearchCV(svr, param_grid, scoring="r2", cv=5)

grid_search.fit(x_train, y_train_scaled)

print("best params - ", grid_search.best_params_)

best_model = SVR(kernel="linear", C=10, epsilon=0.1)

best_model.fit(x_train, y_train_scaled)

y_test_pred_scaled = best_model.predict(x_test)
y_train_pred_scaled = best_model.predict(x_train)

print("train r2: ", r2_score(y_train_scaled, y_train_pred_scaled))
print("test r2: ", r2_score(y_test_scaled, y_test_pred_scaled))



model = LinearSVR(C=10, epsilon=0.1, max_iter=5000)

model.fit(x_train, y_train_scaled)

y_test_pred_scaled = model.predict(x_test)
y_train_pred_scaled = model.predict(x_train)

print("train r2: ", r2_score(y_train_scaled, y_train_pred_scaled))
print("test r2: ", r2_score(y_test_scaled, y_test_pred_scaled))