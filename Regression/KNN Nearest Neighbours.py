# Import Modules
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score)

# Load Data
heart_df = pd.read_csv("heart.csv")
x = heart_df.drop("target",axis = 1)
y = heart_df["target"]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# K = 7
knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(x_train_scaled,y_train)

y_pred = knn_classifier.predict(x_test_scaled)

print("Recall Score : ",recall_score(y_test,y_pred))
print("Accuracy Score : ",accuracy_score(y_test,y_pred))
print("Precision Score : ",precision_score(y_test,y_pred))

# Cross Validation for Hyperparam tuning using GridSearchCV
classifier = KNeighborsClassifier()
param_grid = {"n_neighbors": [3,5,7,9]}

classifiercv = GridSearchCV(
    classifier,
    param_grid,
    cv = 5,
    scoring="recall"
)

classifiercv.fit(x_train_scaled,y_train)
y_pred = classifiercv.predict(x_test_scaled)

print("Recall Score : ",recall_score(y_test,y_pred))
print("Accuracy Score : ",accuracy_score(y_test,y_pred))
print("Precision Score : ",precision_score(y_test,y_pred))

#Results
res = pd.DataFrame(classifiercv.cv_results_)
print(res[["param_n_neighbors","mean_test_score"]])
print(classifiercv.best_params_)