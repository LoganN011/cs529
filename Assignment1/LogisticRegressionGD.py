import numpy as np

class LogisticRegressionGD:

    def __init__(self, eta=0.01, n_iter=50, random_state=1,batch_size=32):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.batch_size = batch_size

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

    def fit_mini_batch_SGD(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.losses_ = []

        n_samples = X.shape[0]

        X_with_bias = np.insert(X, 0, 1, axis=1)

        for _ in range(self.n_iter):
            indices = rgen.permutation(n_samples)
            X_shuffled = X_with_bias[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                xi = X_shuffled[i:i + self.batch_size]
                yi = y_shuffled[i:i + self.batch_size]

                output = self.activation(self.net_input(xi))
                errors = (yi - output)

                self.w_ += self.eta * 2.0 * xi.T.dot(errors) / xi.shape[0]

            full_output = self.activation(self.net_input(X_with_bias))
            loss = (-y.dot(np.log(full_output)) - ((1 - y).dot(np.log(1 - full_output))))
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_)

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        X_with_bias = np.insert(X, 0, 1, axis=1)
        return np.where(self.activation(self.net_input(X_with_bias)) >= 0.5, 1, 0)
