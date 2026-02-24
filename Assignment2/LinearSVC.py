import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Classification import make_classification
from Utils import plot_decision_regions

def get_scaled_iris():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)
    X = df.iloc[:100, [0, 2]].values
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    return X, y


class LinearSVC:
    def __init__(self, eta=0.01, n_iter=50, random_state=1, C=1.0):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.C = C

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.0
        self.losses_ = []

        y_copy = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iter):
            indices = np.arange(X.shape[0])
            rgen.shuffle(indices)
            epoch_loss = 0

            for i in indices:
                val = y_copy[i] * self.net_input(X[i])

                if val >= 1:
                    self.w_ -= self.eta * (2 * (1 / self.C) * self.w_)
                else:
                    self.w_ -= self.eta * (2 * (1 / self.C) * self.w_ - np.dot(y_copy[i], X[i]))
                    self.b_ -= self.eta * -y_copy[i]

                epoch_loss += max(0, 1 - val)

            l2_term = (1 / self.C) * np.dot(self.w_, self.w_)
            self.losses_.append(epoch_loss + l2_term)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


if __name__ == "__main__":
    X,X_test, Y, Y_test = make_classification(d=2, n=100, u=10,random_state=1)
    X_combined = np.vstack((X, X_test))
    y_combined = np.hstack((Y, Y_test))
    svc = LinearSVC(eta=0.0001, n_iter=1000, C=10.0)
    svc.fit(X, Y)
    plot_decision_regions(X_combined, y_combined,classifier=svc,test_idx=range(len(X),len(X_combined)))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


#Task 3 and 4
