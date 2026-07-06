import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------------------------------------------------------------
# 1. Load the Iris dataset
# ------------------------------------------------------------------
iris = load_iris()
x = pd.DataFrame(iris.data)  # type: ignore   # 4 features: sepal/petal length & width
y = iris.target               # type: ignore   # species labels (0, 1, 2) — used only for coloring the plot

# ------------------------------------------------------------------
# 2. Standardize features (mean=0, std=1)
# ------------------------------------------------------------------
# PCA is variance-based, so features with larger raw scales would
# dominate the principal components unless everything is scaled first.
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print(x_scaled.shape)  # (150, 4) — 150 samples, 4 original features

# ------------------------------------------------------------------
# 3. Apply PCA to reduce 4 features down to 2 components
# ------------------------------------------------------------------
# n_components=2 lets us visualize the high-dimensional data in a 2D plot.
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

# explained_variance_ratio_ tells us how much of the original variance
# each principal component captures (e.g. [0.72, 0.23] means PC1 captures
# 72% of the total variance, PC2 captures 23%).
print(pca.explained_variance_ratio_)

# components_ shows how each original feature contributes to