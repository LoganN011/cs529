from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from Utils import plot_decision_regions

def make_classification(d,n,u,random_state=None,return_line=False):
    rgen = np.random.RandomState(random_state)
    a = rgen.randn(d)

    X = rgen.uniform(-u, u, (n, d))

    projections = np.dot(X, a)
    y = np.where(projections < 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=random_state
    )
    if return_line:
        return X_train, X_test, y_train, y_test, a
    else:
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    X_train, X_test, y_train, y_test,a = make_classification(d=2, n=100, u=10,random_state=1,return_line=True)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(X_combined, y_combined,test_idx=range(len(X_train),len(X_combined)))
    x1_min, x1_max = X_combined[:, 0].min() - 1, X_combined[:, 0].max() + 1
    x1_values = np.array([x1_min, x1_max])
    x2_values = -(a[0] / a[1]) * x1_values
    plt.plot(x1_values, x2_values, 'k--', label='Decision Boundary')
    plt.ylim(X_combined[:, 1].min() - 1, X_combined[:, 1].max() + 1)
    plt.legend(loc='upper left')
    plt.show()

