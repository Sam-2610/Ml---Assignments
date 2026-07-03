# Elbow Method
wcss = []
for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit_predict(x)
    wcss.append(kmeans.inertia_)

sns.lineplot(x=range(1, 21), y=wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("WCSS")
plt.show()

# Kneed Module
from kneed import KneeLocator

knee = KneeLocator(
    range(1, 21),
    wcss,
    curve="convex",
    direction="decreasing"
)
print("Optimal k = ", knee.elbow)

# Silhouette Score
from sklearn.metrics import silhouette_score

ss = []
for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(x)
    score = silhouette_score(x, labels)
    ss.append(score)

sns.lineplot(x=range(2, 21), y=ss, marker='o')
plt.title("Silhouette Score by k")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.show()