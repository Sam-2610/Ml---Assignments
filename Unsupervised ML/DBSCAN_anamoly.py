import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Generate a synthetic "two moons" dataset — classic non-linear clustering example
# X = feature coordinates, y = true labels (not used here since clustering is unsupervised)
X, y = make_moons(
    n_samples=500,
    noise=0.1,
    random_state=42  # ensures reproducible results
)

# Standardize features to mean=0, std=1 — important for DBSCAN since it's distance-based
# (eps is a distance threshold, so unscaled features with different ranges would distort it)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Visualize the raw (unclustered) scaled data
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1])

# DBSCAN: density-based clustering
# eps = max distance between two points to be considered neighbors
# min_samples = minimum points required to form a dense region (core point)
dbscan = DBSCAN(
    eps=0.18,
    min_samples=5
)

# fit_predict runs clustering and returns a cluster label for each point
# label -1 means the point was classified as noise (didn't fit any cluster)
labels = dbscan.fit_predict(X_scaled)

# Visualize the clustered result — use `hue`, not `c`, to color points by cluster label in seaborn
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette="deep")