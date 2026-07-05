import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
import seaborn as sns

# ------------------------------------------------------------------
# 1. Load and prepare the Iris dataset
# ------------------------------------------------------------------
iris = load_iris()
x = iris.data       # type: ignore   # feature matrix (150 samples, 4 features)
y = iris.target     # type: ignore   # true species labels (not used for clustering, just for reference)

# Standardize features (mean=0, std=1) since DBSCAN/KMeans are distance-based
# and features on different scales would otherwise dominate the distance calc
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Quick look at two of the four features before clustering
plt.figure()
sns.scatterplot(x=x_scaled[:, 0], y=x_scaled[:, 2])
plt.title("Iris - raw scatter (sepal length vs petal length, scaled)")
plt.show()

# ------------------------------------------------------------------
# 2. DBSCAN on Iris
# ------------------------------------------------------------------
# eps is the max distance between two points for them to be considered neighbors.
# eps=0.0 was a bug — it means NO point can ever be a neighbor of another,
# so every point gets labeled as noise (-1). Using a realistic value instead.
dbscan = DBSCAN(
    eps=0.5,       # fixed: was 0.0
    min_samples=5  # min points required to form a dense region (a cluster)
)
labels = dbscan.fit_predict(x_scaled)

plt.figure()
# hue= (not c=) tells seaborn these are discrete categories, giving a proper legend
sns.scatterplot(x=x_scaled[:, 0], y=x_scaled[:, 2], hue=labels, palette="tab10")
plt.title("Iris - DBSCAN clusters")
plt.show()

# ------------------------------------------------------------------
# 3. Generate non-linear data (two interleaving crescents/moons)
# ------------------------------------------------------------------
# This dataset is a classic example of clusters that are NOT linearly separable,
# useful for showing where KMeans struggles and DBSCAN shines.
x, y = make_moons(
    n_samples=300,
    noise=0.05,      # small amount of jitter so points aren't perfectly on the curve
    random_state=42  # fixed seed for reproducibility
)
x_scaled = scaler.fit_transform(x)  # re-scale this new dataset (scaler is refit here)

plt.figure()
sns.scatterplot(x=x_scaled[:, 0], y=x_scaled[:, 1])
plt.title("Make Moons - raw scatter")
plt.show()

# ------------------------------------------------------------------
# 4. K-Means on the moons data
# ------------------------------------------------------------------
# K-Means assumes roughly spherical, evenly-sized clusters, so it typically
# fails to correctly separate the two crescent shapes (worth showing as a contrast to DBSCAN)
kmeans = KMeans(
    n_clusters=2,
    random_state=42
)
labels = kmeans.fit_predict(x_scaled)

plt.figure()
sns.scatterplot(x=x_scaled[:, 0], y=x_scaled[:, 1], hue=labels, palette="tab10")
plt.title("Make Moons - KMeans clusters")
plt.show()

# ------------------------------------------------------------------
# 5. DBSCAN on the moons data
# ------------------------------------------------------------------
# DBSCAN groups points by density rather than assuming a shape,
# so it should correctly separate the two crescents where KMeans failed
dbscan = DBSCAN(
    eps=0.3,        # tighter eps than before since this dataset is denser/smaller scale
    min_samples=5
)
labels = dbscan.fit_predict(x_scaled)

plt.figure()
sns.scatterplot(x=x_scaled[:, 0], y=x_scaled[:, 1], hue=labels, palette="tab10")
plt.title("Make Moons - DBSCAN clusters")
plt.show()