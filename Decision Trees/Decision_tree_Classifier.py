# Import Modules
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree   # Bug 3 fixed
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
titanic_dataset = sns.load_dataset("titanic")
features = ["pclass", "sex", "fare", "embarked", "age"]
target = "survived"

# ─────────────────────────────────────────────
# 2. Handle Missing Values
# ─────────────────────────────────────────────
# Fill missing age with median (robust to outliers)
imp_median = SimpleImputer(strategy="median")
titanic_dataset[["age"]] = imp_median.fit_transform(titanic_dataset[["age"]])

# Fill missing embarked with most frequent port
imp_freq = SimpleImputer(strategy="most_frequent")
titanic_dataset[["embarked"]] = imp_freq.fit_transform(titanic_dataset[["embarked"]])

# ─────────────────────────────────────────────
# 3. Encode Categorical Columns
# ─────────────────────────────────────────────
le = LabelEncoder()
# Bug 2 fixed — LabelEncoder needs 1D arrays (single brackets)
titanic_dataset["sex"]      = le.fit_transform(titanic_dataset["sex"]) # type: ignore
titanic_dataset["embarked"] = le.fit_transform(titanic_dataset["embarked"]) # type: ignore

# ─────────────────────────────────────────────
# 4. Prepare Features and Target
# ─────────────────────────────────────────────
x = titanic_dataset[features]
y = titanic_dataset[target]

# ─────────────────────────────────────────────
# 5. Train Test Split
# ─────────────────────────────────────────────
# Bug 1 fixed — correct order: x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 6. Decision Tree — No Pruning (Full Tree)
# ─────────────────────────────────────────────
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Accuracy Score (No Pruning) : ", accuracy_score(y_test, y_pred))

# Plot unpruned tree
plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=x.columns, # type: ignore
    class_names=["Died", "Survived"],
    filled=True
)
plt.title("Decision Tree — No Pruning")
plt.tight_layout()
plt.show()   # Bug 4 fixed — added plt.show()

# ─────────────────────────────────────────────
# 7. Decision Tree — Pre Pruning (max_depth)
# ─────────────────────────────────────────────
# Try different max depths and compare accuracy
max_depths = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for depth in max_depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(x_train, y_train)

    acc = model.score(x_test, y_test)
    print(f"Max Depth = {depth}, Accuracy = {acc:.4f}")

    # Bug 5 fixed — plot at depth == 2 (first valid depth in list)
    if depth == 2:
        plt.figure(figsize=(18, 10))
        plot_tree(
            model,
            feature_names=x.columns, # type: ignore
            class_names=["Died", "Survived"],
            filled=True
        )
        plt.title(f"Decision Tree — max_depth = {depth}")
        plt.tight_layout()
        plt.show()

# ─────────────────────────────────────────────
# 8. Decision Tree — Pre Pruning (min_samples_split)
# ─────────────────────────────────────────────
# Try different min_samples_split values with max_depth=4
min_sample_splits = [10, 15, 20, 25, 30]

for split in min_sample_splits:
    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=split,
        random_state=42
    )
    model.fit(x_train, y_train)   # Bug 6 fixed — fit is inside the loop

    acc = model.score(x_test, y_test)
    print(f"Min Samples Split = {split}, Accuracy = {acc:.4f}")

    # Plot tree for first split value
    if split == 10:
        plt.figure(figsize=(18, 10))
        plot_tree(
            model,
            feature_names=x.columns, # type: ignore
            class_names=["Died", "Survived"],
            filled=True
        )
        plt.title(f"Decision Tree — min_samples_split = {split}")
        plt.tight_layout()
        plt.show()

# ─────────────────────────────────────────────
# 9. Decision Tree — Post Pruning (ccp_alpha)
# ─────────────────────────────────────────────
# Train full unpruned tree to extract effective alphas
full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(x_train, y_train)

# Bug 7 fixed — extract ccp_alphas from the pruning path (was undefined as cpp_alphas)
path = full_tree.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas = path.ccp_alphas

# Train one tree per alpha value
trees = []
for alpha in ccp_alphas:
    model = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    model.fit(x_train, y_train)
    trees.append((model, alpha))

# Find the alpha that gives the best test accuracy
best_acc   = 0
best_alpha = 0

for model, alpha in trees:
    curr_acc = model.score(x_test, y_test)
    if curr_acc > best_acc:
        best_acc   = curr_acc
        best_alpha = alpha

print(f"\nBest Alpha : {best_alpha:.6f}")
print(f"Best Accuracy : {best_acc:.4f}")

# Train final model with best alpha and max_depth=4
best_model = DecisionTreeClassifier(ccp_alpha=best_alpha, max_depth=4, random_state=42)
best_model.fit(x_train, y_train)

# Plot the best post-pruned tree
plt.figure(figsize=(18, 10))
plot_tree(
    best_model,
    feature_names=x.columns, # type: ignore
    class_names=["Died", "Survived"],
    filled=True
)
plt.title(f"Decision Tree — Post Pruning (ccp_alpha = {best_alpha:.4f})")
plt.tight_layout()
plt.show()

print("Best Model Accuracy : ", best_model.score(x_test, y_test))