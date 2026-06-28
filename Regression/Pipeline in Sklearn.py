# Import Modules
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score)

# Load Dataset
heart_df = pd.read_csv("heart.csv")
x = heart_df.drop("target", axis=1)
y = heart_df["target"]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_grid = {'knn__n_neighbors': [3, 5, 7, 9]}

# GridSearchCV
classifiercv = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring="recall"
)

classifiercv.fit(x_train, y_train)
y_pred = classifiercv.predict(x_test)

# Evaluation
print("Recall Score : ", recall_score(y_test, y_pred))
print("Accuracy Score : ", accuracy_score(y_test, y_pred))
print("Precision Score : ", precision_score(y_test, y_pred))