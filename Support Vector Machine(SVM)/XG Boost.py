import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic dataset
x, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=10,
    n_redundant=2,
    random_state=42
)

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# Initialize the XGBoost Classifier
# Fixed typo: 'logless' changed to 'logloss'
xgb_elf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    eval_metric="logloss", 
    random_state=42
)

# Fit the model
xgb_elf.fit(x_train, y_train)

# Generate predictions using the test data (This step was missing)
y_pred = xgb_elf.predict(x_test)

# Evaluate the model
print("Accuracy Score : ", accuracy_score(y_test, y_pred))
print("Classification Report :\n", classification_report(y_test, y_pred))