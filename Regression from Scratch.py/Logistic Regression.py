import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iter=1000):
        self.bias = None
        self.weights = None
        self.lr = learning_rate
        self.n_iter = n_iter

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def get_probabilities(self, x):
        z = self.bias + np.dot(x, self.weights) # type: ignore
        return self.__sigmoid(z)

    def fit(self, x, y):
        m, n = x.shape
        self.bias = 0
        self.weights = np.zeros(n)

        for i in range(self.n_iter):
            probabilities = self.get_probabilities(x)

            # Gradients
            db = (1 / m) * np.sum(probabilities - y)
            dw = (1 / m) * np.dot(x.T, (probabilities - y))

            # Update parameters
            self.bias -= self.lr * db
            self.weights -= self.lr * dw

    def predict(self, x, threshold=0.5):
        probabilities = self.get_probabilities(x)
        y_pred_bool = probabilities >= threshold
        return y_pred_bool.astype(int)


if __name__ == "__main__":
    x = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression()
    model.fit(x, y)

    y_pred = model.predict(x)
    print(y_pred)
    print(model.get_probabilities(x))