import numpy as np


class LogisticRegressionGD:

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.losses_ = []

        X_with_bias = np.insert(X, 0, 1, axis=1)
        for i in range(self.n_iter):
            net_input = self.net_input(X_with_bias)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X_with_bias.T.dot(errors) / X.shape[0]
            loss = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))) / X.shape[0])
            self.losses_.append(loss)
        return self

    def  fit_mini_batch_SGD(self, X, y):
        #todo write this function
        return self

    def net_input(self, X):
        return np.dot(X, self.w_)

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        X_with_bias = np.insert(X, 0, 1, axis=1)
        return np.where(self.activation(self.net_input(X_with_bias)) >= 0.5, 1, 0)
