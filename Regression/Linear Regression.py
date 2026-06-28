import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
 
# ──────────────────────────────────────────────
# 1. Load Dataset
# ──────────────────────────────────────────────
 
insurance_data = pd.read_csv("insurance.csv")
print(insurance_data.head())
 
# ──────────────────────────────────────────────
# 2. Visualize Data
# ──────────────────────────────────────────────
 
sns.scatterplot(
    x=insurance_data["bmi"],
    y=insurance_data["charges"],
    hue=insurance_data["smoker"]
)
plt.title("BMI vs Charges (colored by Smoker)")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.show()
 
# ──────────────────────────────────────────────
# 3. Feature Engineering
# ──────────────────────────────────────────────
 
x = insurance_data.drop(columns=["charges"])
y = insurance_data["charges"]
 
# Label Encoding — convert binary categories to 0/1
x["sex"]    = x["sex"].map({"female": 1, "male": 0})
x["smoker"] = x["smoker"].map({"yes": 1, "no": 0})
 
# One-Hot Encoding — convert 'region' into separate columns
x = pd.get_dummies(x, columns=["region"], drop_first=False, dtype=int)
 
# Interaction Features — capture combined effects
x["age_smoker"] = x["age"] * x["smoker"]
x["bmi_smoker"] = x["bmi"] * x["smoker"]
 
print("\nFeatures after engineering:")
print(x.head())
 
# ──────────────────────────────────────────────
# 4. Train-Test Split
# ──────────────────────────────────────────────
 
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
 
print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")
 
# ──────────────────────────────────────────────
# 5. Train Model
# ──────────────────────────────────────────────
 
model = LinearRegression()
model.fit(X_train, y_train)
 
# ──────────────────────────────────────────────
# 6. Predict
# ──────────────────────────────────────────────
 
y_pred       = model.predict(X_test)
y_train_pred = model.predict(X_train)
 
print("\nPredicted charges (first 5):", y_pred[:5])
 
# ──────────────────────────────────────────────
# 7. Model Evaluation
# ──────────────────────────────────────────────
 
# R² Score
r2_test  = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, y_train_pred)
 
print(f"\nTraining R²  : {r2_train:.4f}")
print(f"Test R²      : {r2_test:.4f}")
 
# Adjusted R²
n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r2 = 1 - ((1 - r2_test) * (n - 1) / (n - p - 1))
print(f"Adjusted R²  : {adjusted_r2:.4f}")
 
# Overfitting Check
print("\n── Model Checking ──")
if r2_train - r2_test > 0.1:
    print("⚠ Possible Overfitting: Training R² is much higher than Test R²")
elif r2_test < 0.5:
    print("⚠ Possible Underfitting: Test R² is too low")
else:
    print("✓ Model is generalizing well")