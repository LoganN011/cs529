import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

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
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size= X.shape[1])
        self.b_ = np.float64(0.)
        self.losses_ = []

        y_copy = np.where(y <= 0, -1, 1)
        for _ in range(self.n_iter):
            indices = np.arange(X.shape[0])
            rgen.shuffle(indices)

            loss = 0
            for i in indices:
                condition = y_copy[i] * self.net_input(X[i]) >= 1

                if condition:
                    self.w_ -= self.eta * (1 / self.C * self.w_)
                else:
                    self.w_ += self.eta * (y_copy[i] * X[i] - (1 / self.C * self.w_))
                    self.b_ += self.eta * y_copy[i]

                loss += max(0, 1 - y_copy[i] * self.net_input(X[i]))

            self.losses_.append(loss)


        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

X, Y = get_scaled_iris()
svc = LinearSVC(eta=0.1, n_iter=500)
svc.fit(X, Y)
plot_decision_regions(X, Y, classifier=svc)
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.title('SVC - Iris')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()