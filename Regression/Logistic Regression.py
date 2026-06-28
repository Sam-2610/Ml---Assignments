# Import Modules
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, confusion_matrix,
                             classification_report, recall_score, f1_score)

# Load Data
heart_df = pd.read_csv("heart.csv")
x = heart_df.drop("target", axis=1)
y = heart_df["target"]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Baseline Logistic Regression (unscaled)
model = LogisticRegression(max_iter=100)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Accuracy Score  : ", accuracy_score(y_test, y_pred))
print("Precision Score : ", precision_score(y_test, y_pred))
print("Recall Score    : ", recall_score(y_test, y_pred))
print("F1 Score        : ", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Logistic Regression (scaled)
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)

print("Accuracy Score  : ", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Precision Score : ", round(precision_score(y_test, y_pred) * 100, 2), "%")

# KNN with GridSearchCV
classifier = KNeighborsClassifier()
param_grid = {"n_neighbors": [3, 5, 7, 9]}

classifier_cv = GridSearchCV(classifier, param_grid, cv=5, scoring="recall")
classifier_cv.fit(x_train_scaled, y_train)
y_pred = classifier_cv.predict(x_test_scaled)

print("Recall Score    : ", recall_score(y_test, y_pred))
print("Accuracy Score  : ", accuracy_score(y_test, y_pred))
print("Precision Score : ", precision_score(y_test, y_pred))

res = pd.DataFrame(classifier_cv.cv_results_)
print(res[["param_n_neighbors", "mean_test_score"]])
print("Best Params     : ", classifier_cv.best_params_)