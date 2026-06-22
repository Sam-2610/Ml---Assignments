from sklearn.linear_model import Lasso, LassoCV          # ✅ capital L
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
insurance_data = pd.read_csv("insurance.csv")
x = insurance_data.drop(columns=["charges"])
y = insurance_data["charges"]

# Encode categorical columns
x = pd.get_dummies(x, columns=["region"], drop_first=True, dtype=int)
x["sex"]    = x["sex"].map({"female": 1, "male": 0})
x["smoker"] = x["smoker"].map({"yes": 1, "no": 0})

# Feature engineering
x["age_smoker"] = x["age"] * x["smoker"]
x["bmi_smoker"] = x["bmi"] * x["smoker"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Manual alpha search
alphas = [0.001, 0.1, 1, 2, 5, 10, 25]
mses = []                                                 # ✅ keep as list

for a in alphas:
    lasso_model = Lasso(alpha=a)                          # ✅ Lasso, alpha
    lasso_model.fit(x_train, y_train)

    y_pred = lasso_model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)              # ✅ separate variable
    print(f"MSE for alpha = {a} : {mse}")
    mses.append(mse)                                      # ✅ append to list

# Plot
sns.lineplot(x=alphas, y=mses, marker="o")               # ✅ "o" not "0"
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.title("Lasso Alpha vs MSE")
plt.show()

# LassoCV - auto finds best alpha
lasso_cv_model = LassoCV(
    alphas=[0.001, 0.1, 1, 2, 5, 10],
    cv=5,
    max_iter=1000,
    random_state=42
)
lasso_cv_model.fit(x_train, y_train)

print("Best Alpha:", lasso_cv_model.alpha_)

y_pred = lasso_cv_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print("MSE :", mse)
print("R2  :", r2)                                        # ✅ fixed typo