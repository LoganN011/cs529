import numpy as np
from AdalineGD import AdalineGD

class MultiClassAdalineGD:

    def __init__(self, eta=0.01, n_iter=1000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        y_model1 = np.where(y == 0, 1, 0)
        self.model1_ = AdalineGD(n_iter=self.n_iter, eta=self.eta, random_state=self.random_state)
        self.model1_.fit(X, y_model1)

        mask = (y == 1) | (y == 2)
        X_sub = X[mask]
        y_sub = y[mask]

        y_model2 = np.where(y_sub == 1, 1, 0)
        self.model2_ = AdalineGD(n_iter=self.n_iter, eta=self.eta, random_state=self.random_state)
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