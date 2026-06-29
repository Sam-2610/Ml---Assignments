import numpy as np

# Linear Regression
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.bias = None
        self.weights = None
        self.lr = learning_rate
        self.n_iter = n_iter

    def fit(self, x, y):
        m, n = x.shape

        # Step 1 : Initialise
        self.bias = 0
        self.weights = np.zeros(n)

        # Gradient descent
        for i in range(self.n_iter):
            y_pred = self.bias + np.dot(x, self.weights)

            # Step 3 : Calculate gradient
            db = (1 / m) * np.sum(y_pred - y)
            dw = (1 / m) * np.dot(x.T, (y_pred - y))

            # Step 4 : Update parameters
            self.bias -= self.lr * db
            self.weights -= self.lr * dw

    def predict(self, x):
        return self.bias + np.dot(x, self.weights) # type: ignore


if __name__ == "__main__":
    x = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    model = LinearRegression(learning_rate=0.01, n_iter=1000)
    model.fit(x, y)
    y_pred = model.predict(x)
    print(y_pred)