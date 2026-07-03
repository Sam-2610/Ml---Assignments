from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

iris = load_iris()
x = iris.data
y = iris.target

# Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Visualise raw (scaled) data
plt.figure(figsize=(6, 5))
sns.scatterplot(x=x_scaled[:, 0], y=x_scaled[:, 2], hue=y, palette="deep")
plt.title("Scaled Iris Features")
plt.show()

# Linkage matrix
z = linkage(x_scaled, method="ward")

# Dendrogram
plt.figure(figsize=(12, 6))
dendrogram(z)
plt.xlabel("samples")
plt.ylabel("distances")
plt.title("Dendrogram for Hierarchical Clustering")
plt.show()

# Clustering
agg = AgglomerativeClustering(n_clusters=3)
labels = agg.fit_predict(x_scaled)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=x_scaled[:, 0], y=x_scaled[:, 2], hue=labels, palette="deep")
plt.title("Agglomerative Clustering Results")
plt.show()