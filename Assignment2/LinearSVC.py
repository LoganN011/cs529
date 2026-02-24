import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Classification import make_classification
from sklearn.svm import LinearSVC as sklearnSVC
from Utils import plot_decision_regions


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
            avg_epoch_loss = (epoch_loss / X.shape[0]) + l2_term
            self.losses_.append(avg_epoch_loss)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

def run_scalability_test(model, d_list, n_list, u_val=100):
    results = []

    for d in d_list:
        for n in n_list:
            print(f"Testing: n={n}, d={d}...")


            X, X_test, y,y_test = make_classification(n=n, d=d, u=u_val,random_state=1)
            X =np.vstack((X, X_test))
            y =np.hstack((y, y_test))

            start_time = time.time()
            model.fit(X, y)
            end_time = time.time()
            elapsed_time = end_time - start_time

            plt.plot(range(1, len(model.losses_) + 1), model.losses_, linestyle='-', color='blue')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Scalability Test: D={d}, N={n}')
            plt.tight_layout()
            plt.show()

            results.append({
                'Samples (n)': n,
                'Dimensions (d)': d,
                'Time (s)': round(elapsed_time, 4)
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # X,X_test, Y, Y_test = make_classification(d=2, n=100, u=10,random_state=1)
    # X_combined = np.vstack((X, X_test))
    # y_combined = np.hstack((Y, Y_test))
    # svc = LinearSVC(eta=0.0001, n_iter=1000, C=10.0)
    # svc.fit(X, Y)
    # plot_decision_regions(X_combined, y_combined,classifier=svc,test_idx=range(len(X),len(X_combined)))
    # plt.legend(loc='upper left')
    # plt.tight_layout()
    # plt.show()


    #Task 3 and 4


    # --- Execution ---
    d_scales = [10, 50, 100]
    n_scales = [50, 500, 5000]
    #Task 3
    # svc = LinearSVC(eta=0.00001, n_iter=1000, C=1.0)
    # df_results = run_scalability_test(svc, d_scales, n_scales)
    #
    #
    # pivot_table = df_results.pivot(index='Samples (n)', columns='Dimensions (d)', values='Time (s)')
    # print("\n--- Time Cost (Seconds) ---")
    # print(pivot_table)

    #Task 4
    dual = sklearnSVC(dual=True)
    df_results = run_scalability_test(dual, d_scales, n_scales)


    pivot_table = df_results.pivot(index='Samples (n)', columns='Dimensions (d)', values='Time (s)')
    print("\n--- Time Cost (Seconds) ---")
    print(pivot_table)

    primal = sklearnSVC(dual=False)


