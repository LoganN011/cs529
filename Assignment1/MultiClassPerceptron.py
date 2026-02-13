import numpy as np

class Perceptron:

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

class MultiClassPerceptron:

    def __init__(self, eta=0.01, n_iter=1000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        y_model1 = np.where(y == 0, 1, 0)
        self.model1_ = Perceptron(n_iter=self.n_iter, eta=self.eta, random_state=self.random_state)
        self.model1_.fit(X, y_model1)

        mask = (y == 1) | (y == 2)
        X_sub = X[mask]
        y_sub = y[mask]

        y_model2 = np.where(y_sub == 1, 1, 0)
        self.model2_ = Perceptron(n_iter=self.n_iter, eta=self.eta, random_state=self.random_state)
        self.model2_.fit(X_sub, y_model2)

        return self

    def predict(self, X):
        pred1 = self.model1_.predict(X)

        prediction = np.zeros(X.shape[0])

        for i in range(len(pred1)):
            if pred1[i] == 1:
                prediction[i] = 0
            else:
                pred2 = self.model2_.predict(X[i].reshape(1, -1))
                prediction[i] = 1 if pred2 == 1 else 2

        return prediction