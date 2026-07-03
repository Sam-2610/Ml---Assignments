import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
x = iris.data
y = iris.target

# Visualise raw features
plt.figure(figsize=(6, 5))
sns.scatterplot(x=x[:, 0], y=x[:, 2], hue=y, palette="deep")
plt.title("Raw Iris Features")
plt.show()

# Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Dimensionality reduction using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(x_scaled)

# Elbow Method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pca_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6, 5))
sns.lineplot(x=range(1, 11), y=wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("WCSS")
plt.show()

# KMeans with chosen k
kmeans = KMeans(n_clusters=3, random_state=10, n_init=10)
labels = kmeans.fit_predict(pca_data)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=labels, palette="deep")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X", c="red", s=200, label="Centroids"
)
plt.title("KMeans Clusters (PCA-reduced)")
plt.legend()
plt.show()