from sklearn.neighbors import LocalOutlierFactor

# LocalOutlierFactor (LOF): another unsupervised anomaly detection method
# Unlike IsolationForest, LOF measures local density deviation —
# a point is flagged as an outlier if it has significantly lower density than its neighbors
# contamination = expected proportion of outliers (same role as in IsolationForest)
neighbs = LocalOutlierFactor(contamination=0.036)

# fit_predict returns 1 for normal points, -1 for outliers
# Note: LOF has no separate .fit() then .predict() — fit_predict is the only supported way
# to get labels for the training data itself (no separate .predict() for new/unseen points)
labels = neighbs.fit_predict(X_scaled)

plt.figure(figsize=(8, 6))

# Reusing the same PCA-reduced coordinates from before, just recoloring by LOF's labels
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()  # needed to actually render the plot in a plain script

# Count how many points LOF flagged as outliers (-1) vs normal (1)
n_outliers = np.sum(labels == -1)
n_normal = np.sum(labels == 1)

print("outlier = ", n_outliers)
print("normal = ", n_normal)