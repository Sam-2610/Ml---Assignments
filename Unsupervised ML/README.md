# Unsupervised Learning — Clustering & Anomaly Detection

A collection of standalone Python scripts exploring clustering and anomaly detection techniques using scikit-learn, applied to the Iris dataset, synthetic data (make_moons), and a thyroid dataset.

## Overview

This repo demonstrates and compares several unsupervised learning algorithms:

- **Clustering** — grouping similar data points together (K-Means, DBSCAN, Agglomerative/Hierarchical)
- **Anomaly Detection** — identifying outliers/unusual points (Isolation Forest, Local Outlier Factor, DBSCAN)
- **Dimensionality Reduction** — PCA, used both for visualization and as a preprocessing step before clustering

## Files

| File | Description |
|---|---|
| `PCA.py` | Introduces Principal Component Analysis on the Iris dataset — reduces 4 features to 2 components for visualization, explains variance captured by each component. |
| `K_-_Means.py` | K-Means model selection: Elbow Method (WCSS) with automatic elbow detection via `kneed`, plus Silhouette Score analysis across a range of k values. |
| `kMeans_for_iris.py` | Full K-Means pipeline on Iris: scaling → PCA (2D) → elbow method → final clustering with centroids plotted. |
| `Hierachical_Clustering.py` | Agglomerative (hierarchical) clustering on Iris: dendrogram visualization via `scipy.cluster.hierarchy`, followed by clustering with `AgglomerativeClustering` (ward linkage). |
| `DBSCAN.py` | Compares DBSCAN vs. K-Means on both Iris and the synthetic "two moons" dataset — demonstrates how DBSCAN handles non-linearly-separable clusters that K-Means fails on. |
| `DBSCAN_anamoly.py` | Uses DBSCAN's noise classification (label `-1`) as an anomaly detection method on the "two moons" dataset. |
| `Isolation_Forest.py` | Anomaly detection on a thyroid dataset using `IsolationForest`, with PCA-based 2D visualization of flagged outliers vs. normal points. |
| `LOF.py` | Anomaly detection using `LocalOutlierFactor` (density-based) on the same thyroid dataset/PCA projection, for comparison against Isolation Forest. |

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
kneed
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy kneed
```

## Datasets Used

- **Iris** — loaded directly via `sklearn.datasets.load_iris()`, no external file needed.
- **Make Moons** — synthetic data generated via `sklearn.datasets.make_moons()`, no external file needed.
- **`thyroid_dataset.csv`** — required locally in the working directory for `Isolation_Forest.py` and `LOF.py`. Must contain an `Outlier_label` column (used only for reference, not for training, since these are unsupervised methods).

## Usage

Each standalone script can be run independently:

```bash
python PCA.py
python kMeans_for_iris.py
python Hierachical_Clustering.py
python DBSCAN.py
python DBSCAN_anamoly.py
python Isolation_Forest.py
```

## Important Notes

- **`LOF.py` and `K_-_Means.py` are continuation snippets, not standalone scripts.** They reference variables (`X_scaled`, `X_pca`, `x`, `plt`, `np`, `KMeans`, `sns`) that aren't defined or imported within the file itself. `LOF.py` is meant to run immediately after `Isolation_Forest.py` in the same session (reusing its PCA projection). `K_-_Means.py` similarly expects `x`, `KMeans`, `sns`, and `plt` to already be defined — likely intended to run after a script like `kMeans_for_iris.py`'s scaling/PCA steps. Run these together with their companion script, or add the missing imports/variable definitions if running in isolation.
- **`PCA.py` is incomplete** — the final comment (`# components_ shows how each original feature contributes to`) is cut off mid-sentence, with no corresponding code printing `pca.components_`.
- Random seeds (`random_state=42`) are used throughout for reproducibility — results should be identical across runs.
- Scripts using seaborn's `scatterplot` correctly use `hue=` (not `c=`) for categorical cluster coloring; scripts using matplotlib's `plt.scatter` use `c=` directly, which is valid there.

## Key Takeaways Demonstrated

- **K-Means** assumes roughly spherical, evenly-sized clusters and struggles with non-linear shapes (shown clearly in `DBSCAN.py`'s moons comparison).
- **DBSCAN** clusters by density rather than shape, correctly separating non-linear clusters, and naturally flags low-density points as noise/outliers (label `-1`) — useful for anomaly detection too.
- **Agglomerative Clustering** builds a hierarchy of merges (visualized via dendrogram) and doesn't require specifying k in advance for exploration, though a final cut (`n_clusters`) is still needed to get flat labels.
- **Isolation Forest** and **Local Outlier Factor** offer two different philosophies for anomaly detection — isolation-based (how easily a point can be split off) vs. density-based (how sparse a point's neighborhood is) — and can be compared on the same dataset to see where they agree/disagree.
- **PCA** is used throughout both as a visualization tool (reducing to 2–3 dimensions) and as a preprocessing step before clustering (reducing noise/dimensionality).