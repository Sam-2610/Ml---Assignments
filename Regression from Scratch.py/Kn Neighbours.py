# Import Modules
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# KNN Implementation from scratch for Binary Classification
class KnnClassifier:
    def __init__(self, k=3):
        self.k = k

    def __euclidean_dist(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def __predict_one(self, x):
        distance = [self.__euclidean_dist(x, x_train) for x_train in self.x_train]
        knn_indices = np.argsort(distance)[:self.k]
        knn_classes = [int(self.y_train[i]) for i in knn_indices]
        majority_class = np.argmax(np.bincount(knn_classes))
        return majority_class

    def predict(self, x):
        y_pred = [self.__predict_one(xi) for xi in x]
        return np.array(y_pred)


if __name__ == "__main__":
    x_train = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [6, 5],
        [7, 7]
    ])
    y_train = np.array([0, 0, 0, 1, 1])
    x_test = np.array([
        [2, 2],
        [6, 6]
    ])

    model = KnnClassifier()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(y_pred)

    # Visualize training points
    x_train_df = pd.DataFrame(x_train, columns=["x1", "x2"])
    x_train_df["label"] = y_train

    sns.scatterplot(
        data=x_train_df,
        x="x1",
        y="x2",
        hue="label"
    )

    # Visualize test points
    x_test_df = pd.DataFrame(x_test, columns=["x1", "x2"])

    sns.scatterplot(
        data=x_test_df,
        x="x1",
        y="x2",
        color="black",
        marker="x",
        label="Test Points"
    )

    plt.title("KNN — Training vs Test Points")
    plt.show()