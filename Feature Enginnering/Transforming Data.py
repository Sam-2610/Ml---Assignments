# Transforming Data (Feature Engineering)

import pandas as pd

# Create sample DataFrame
df = pd.DataFrame({
    'income': [5000, 7000, 6000, 8000, 5500],
    'gender': ['male', 'Female', 'male', 'Unknown', 'Female'],
    'country': ['USA', 'Canada', 'USA', 'UK', 'Canada'],
    'age': [25, 30, 35, 40, 28]
})

df2 = df.copy()

# Apply()
df2["tax"] = df2["income"].apply(lambda x: "20%" if x >= 6000 else "10%")

# Map()
gender_map = {"male": "M", "Female": "F", "Unknown": "U"}
df2["gender"] = df2["gender"].map(gender_map)

# Assign()
df2 = df2.assign(new_income=df2["income"] * 1.1)

# Replace(old, new)
df2["country"] = df2["country"].replace("USA", "US")

# Rename()
df_example = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [4, 5, 6],
    "C": [7, 8, 9]
})

df_rename = df_example.rename(columns={"A": "X", "B": "Y"})
print(df_rename)

# Reorder Columns
df_reordered = df_example[["C", "A", "B"]]

# Sorting Values and Index
df2.sort_values("income")  # Ascending
df2.sort_values("income", ascending=False)  # Descending
df2.sort_values(["age", "income"])  # sorts age if age same then sorts income

sorted_df2 = df2.sort_values(["age", "income"])
sorted_df2 = sorted_df2.sort_index()

# Reset Index
sorted_df2 = sorted_df2.reset_index(drop=True)  # to drop original index values

# Ranking
df2["Ranking"] = df2["income"].rank(ascending=False, method="dense")
df2["Ranking"] = df2["income"].rank(ascending=False, method="min")
df2["Ranking"] = df2["income"].rank(ascending=False, method="max")

# Groupby()
df2.groupby("country")["income"].mean()  # mean income for each country

# Agg()
df2.groupby("country")["income"].agg(["mean", "min", "max"])

# Rename Aggregate
df2.groupby("country")["income"].agg(avg_salary="mean", max_salary="max")

df2.groupby("country").agg({
    "age": "mean",
    "income": "mean",
})  # agg on multiple columns

df2.groupby("country").agg(
    avg_age=("age", "mean"),
    avg_salary=("income", "mean")
)