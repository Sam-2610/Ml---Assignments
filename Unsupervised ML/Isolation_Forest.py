import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("thyroid_dataset.csv")

# Separate features (X) from the known outlier label (y)
# Note: y isn't actually used below since IsolationForest is unsupervised —
# it's only useful here if you want to compare predictions against ground truth later
X = df.drop("Outlier_label", axis=1)
y = df["Outlier_label"]

# Standardize features to mean=0, std=1 — helps distance/split-based models
# treat all features on a comparable scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.ensemble import IsolationForest

# IsolationForest: unsupervised anomaly detection
# n_estimators = number of trees in the ensemble
# contamination = expected proportion of outliers in the dataset (helps set the decision threshold)
clf = IsolationForest(n_estimators=200, contamination=0.036, random_state=42)

# fit_predict returns 1 for normal points, -1 for detected outliers
labels = clf.fit_predict(X_scaled)

# Visualize
from sklearn.decomposition import PCA

# Reduce to 2 dimensions purely for visualization purposes
# (IsolationForest itself already ran on the full-dimensional X_scaled above)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

# Color points by their outlier/normal label from IsolationForest
# matplotlib's scatter (unlike seaborn) correctly accepts `c` for numeric-coded colors
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()  # needed to actually display the plot outside notebooks

import numpy as np

# Count how many points were flagged as outliers (-1) vs normal (1)
n_outliers = np.sum(labels == -1)
n_normal = np.sum(labels == 1)

print("outlier = ", n_outliers)
print("normal = ", n_normal)