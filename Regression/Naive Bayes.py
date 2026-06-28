# Import Modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score,recall_score,precision_score)

# Load Dataset
heart_df = pd.read_csv("heart.csv")
x = heart_df.drop("target", axis=1)
y = heart_df["target"]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Naive Base Model
gnb_model = GaussianNB()
gnb_model.fit(x_train, y_train)
y_pred = gnb_model.predict(x_test)

# Evaluation
print("Accuracy Score : ",accuracy_score(y_test,y_pred))
print("Recall Score : ",recall_score(y_test,y_pred))
print("Precision Score : ",precision_score(y_test,y_pred))