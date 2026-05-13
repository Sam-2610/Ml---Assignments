# Linear Regression with Feature Engineering — Python
 
An end-to-end **Linear Regression** pipeline built on the Insurance dataset. This program covers data visualization, feature engineering, model training, prediction, and evaluation — all in a single file.
 
---
 
## Libraries Used
 
| Library | Purpose |
|---------|---------|
| `pandas` | Load and manipulate tabular data |
| `seaborn` | Statistical data visualization |
| `matplotlib` | Plotting graphs |
| `scikit-learn` | Model training, splitting, and evaluation |
 
### Install Dependencies
 
```bash
pip install pandas seaborn matplotlib scikit-learn
```
 
---
 
## Dataset
 
**File:** `insurance.csv`
 
A medical insurance dataset with the following columns:
 
| Column | Type | Description |
|--------|------|-------------|
| `age` | int | Age of the person |
| `sex` | string | Gender (`male` / `female`) |
| `bmi` | float | Body Mass Index |
| `children` | int | Number of dependents |
| `smoker` | string | Whether the person smokes (`yes` / `no`) |
| `region` | string | Residential region in the US |
| `charges` | float | Medical insurance charges billed (**target variable**) |
 
> Download the dataset from [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
 
---
 
## Program Flow
 
```
Load Dataset
     ↓
Visualize (BMI vs Charges scatter plot)
     ↓
Feature Engineering
  ├── Label Encoding  (sex, smoker)
  ├── One-Hot Encoding  (region)
  └── Interaction Features  (age_smoker, bmi_smoker)
     ↓
Train-Test Split (80% train / 20% test)
     ↓
Train Linear Regression Model
     ↓
Predict on Test Data
     ↓
Evaluate (R², Adjusted R², Model Checking)
```
 
---
 
## Step-by-Step Explanation
 
---
 
### 1. Load Dataset
 
```python
insurance_data = pd.read_csv("insurance.csv")
print(insurance_data.head())
```
 
Reads the CSV file into a pandas DataFrame and previews the first 5 rows.
 
---
 
### 2. Visualize Data
 
```python
sns.scatterplot(
    x=insurance_data["bmi"],
    y=insurance_data["charges"],
    hue=insurance_data["smoker"]
)
```
 
Plots BMI on the x-axis and insurance charges on the y-axis, with dots colored by smoker status.
 
**What to look for:**
- Smokers tend to have significantly higher charges than non-smokers
- Higher BMI combined with smoking leads to the highest charges
- Two visible clusters — smokers (top) and non-smokers (bottom)
---
 
### 3. Feature Engineering
 
Raw data often contains text columns that ML models can't process directly. Feature Engineering converts them into numbers.
 
#### a) Label Encoding
 
Converts binary text categories to 0 and 1.
 
```python
x["sex"]    = x["sex"].map({"female": 1, "male": 0})
x["smoker"] = x["smoker"].map({"yes": 1, "no": 0})
```
 
| Original | Encoded |
|----------|---------|
| female   | 1       |
| male     | 0       |
| yes      | 1       |
| no       | 0       |
 
#### b) One-Hot Encoding
 
Converts the `region` column (4 unique values) into 4 separate binary columns.
 
```python
x = pd.get_dummies(x, columns=["region"], drop_first=False, dtype=int)
```
 
| region | region_northeast | region_northwest | region_southeast | region_southwest |
|--------|-----------------|-----------------|-----------------|-----------------|
| northeast | 1 | 0 | 0 | 0 |
| southwest | 0 | 0 | 0 | 1 |
 
- `drop_first=False` — keeps all 4 region columns (no dummy variable trap handling here)
- `dtype=int` — stores as integers (0/1) instead of booleans
#### c) Interaction Features
 
Creates new features by multiplying two existing columns to capture their **combined effect**.
 
```python
x["age_smoker"] = x["age"] * x["smoker"]
x["bmi_smoker"] = x["bmi"] * x["smoker"]
```
 
| Feature | Meaning |
|---------|---------|
| `age_smoker` | Age × Smoker — captures how age affects charges **specifically for smokers** |
| `bmi_smoker` | BMI × Smoker — captures how BMI drives up costs **only when smoking** |
 
For a non-smoker, both values are 0. For a smoker aged 45 with BMI 30: `age_smoker = 45`, `bmi_smoker = 30`.
 
---
 
### 4. Train-Test Split
 
```python
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
```
 
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `test_size=0.2` | 20% | 20% of data held out for testing |
| `random_state=42` | 42 | Fixed seed for reproducibility |
 
- **Training set** — the model learns from this
- **Test set** — used to evaluate how well the model generalises to unseen data
---
 
### 5. Train the Model
 
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
 
`LinearRegression()` fits the equation:
 
```
charges = b0 + b1×age + b2×sex + b3×bmi + b4×children + b5×smoker + ... + bn×bmi_smoker
```
 
`model.fit()` finds the best values for all coefficients (b0, b1, ..., bn) that minimize prediction error.
 
---
 
### 6. Predict
 
```python
y_pred       = model.predict(X_test)     # predictions on test set
y_train_pred = model.predict(X_train)    # predictions on training set
```
 
Generates predicted insurance charges for both the test and training sets — needed for model checking.
 
---
 
### 7. Model Evaluation
 
#### R² Score (Coefficient of Determination)
 
Measures how well the model explains the variance in the target variable.
 
```python
r2_test  = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, y_train_pred)
```
 
| R² Value | Interpretation |
|----------|---------------|
| 1.0 | Perfect fit |
| 0.8+ | Good fit |
| 0.5–0.8 | Moderate fit |
| < 0.5 | Poor fit |
| 0 or negative | Model is worse than a flat line |
 
#### Adjusted R²
 
Penalizes adding features that don't improve the model, making it more reliable than plain R².
 
```python
n = X_test.shape[0]   # number of test samples
p = X_test.shape[1]   # number of features
 
adjusted_r2 = 1 - ((1 - r2_test) * (n - 1) / (n - p - 1))
```
 
- Always ≤ R²
- Decreases if you add a useless feature (unlike R² which always increases)
#### Model Checking (Overfitting / Underfitting)
 
```python
if r2_train - r2_test > 0.1:
    print("⚠ Possible Overfitting")
elif r2_test < 0.5:
    print("⚠ Possible Underfitting")
else:
    print("✓ Model is generalizing well")
```
 
| Situation | Training R² | Test R² | Meaning |
|-----------|-------------|---------|---------|
| Overfitting | High (~0.95) | Low (~0.60) | Model memorized training data, fails on new data |
| Underfitting | Low | Low | Model is too simple to learn the pattern |
| Good fit | High | Close to Training R² | Model generalizes well |
 
---
 
## How to Run
 
1. Place `insurance.csv` in the same folder as the script.
2. Run:
```bash
python Linear_Regression.py
```
 
---
 
## Files
 
| File | Description |
|------|-------------|
| `Linear_Regression.py` | Complete merged pipeline |
| `insurance.csv` | Dataset (download from Kaggle) |
 
---
 
## Author
 
**Satyam Sagar**
📧 satyamsagar827@gmail.com