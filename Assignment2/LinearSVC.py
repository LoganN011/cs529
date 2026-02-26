import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Classification import make_classification
from sklearn.svm import LinearSVC as sklearnSVC
from Utils import plot_decision_regions
from sklearn.metrics import hinge_loss


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
            loss = 0

            for i in indices:
                val = y_copy[i] * self.net_input(X[i])

                if val >= 1:
                    self.w_ -= self.eta * (2 * (1 / self.C) * self.w_)
                else:
                    self.w_ -= self.eta * (self.w_ - self.C * y_copy[i] * X[i])
                    self.b_ -= self.eta * (-self.C * y_copy[i])

                loss += max(0, 1 - val)

            l2_term = (1 / 2) * np.dot(self.w_, self.w_)
            epoch_loss = (loss*self.C / X.shape[0]) + l2_term
            self.losses_.append(epoch_loss)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

def run_scalability_test(model, d_list, n_list, folder="test_data"):
    results = []

    num_plots = len(d_list) * len(n_list)
    fig, axes = plt.subplots(len(d_list), len(n_list), figsize=(5 * len(n_list), 4 * len(d_list)))
    if num_plots > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    plot_idx=0
    for d in d_list:
        for n in n_list:
            filename = f"{folder}/data_n{n}_d{d}.npz"

            if not os.path.exists(filename):
                print(f"Skipping: {filename} not found.")
                continue

            data = np.load(filename)
            X, y = data['X'], data['y']


            start_time = time.time()
            model.fit(X, y)
            end_time = time.time()
            elapsed_time = end_time - start_time

            ax = axes_flat[plot_idx]
            ax.plot(range(1, len(model.losses_) + 1), model.losses_, linestyle='-', color='blue')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.set_title(f'Scalability Test: D={d}, N={n}')
            plot_idx += 1
            results.append({
                'Samples (n)': n,
                'Dimensions (d)': d,
                'Time (s)': round(elapsed_time, 4)
            })

    plt.tight_layout()
    plt.show()
    return pd.DataFrame(results)


def run_sklearn_test(d_list, n_list, max_iter=1000, folder="test_data"):
    for is_dual in [True, False]:
        version_name = 'Dual' if is_dual else 'Primal'


        fig, axes = plt.subplots(len(d_list), len(n_list),figsize=(5 * len(n_list), 4 * len(d_list)))
        axes_flat = axes.flatten()
        plot_idx = 0

        for d in d_list:
            for n in n_list:
                filename = f"{folder}/data_n{n}_d{d}.npz"

                if not os.path.exists(filename):
                    print(f"Skipping: {filename} not found.")
                    plot_idx += 1
                    continue

                data = np.load(filename)
                X, y = data['X'], data['y']

                losses = []
                for i in range(1, max_iter + 1):
                    model = sklearnSVC(dual=is_dual, max_iter=i,random_state=1,penalty='l2')
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X, y)

                    decision = model.decision_function(X)
                    loss = hinge_loss(y, decision)
                    losses.append(loss)

                ax = axes_flat[plot_idx]
                color = 'red' if is_dual else 'blue'

                ax.plot(range(1, len(losses) + 1), losses, color=color)
                ax.set_title(f'{version_name}: n={n}, d={d}')
                ax.set_xlabel('Epochs/Iter')
                ax.set_ylabel('Loss')

                plot_idx += 1

        plt.suptitle(f"Scalability Test: {version_name} Version", fontsize=16)
        plt.tight_layout()
        plt.show()

def run_sklearn_time_test(model,d_list, n_list,folder="test_data"):
    results = []

    for d in d_list:
        for n in n_list:
            filename = f"{folder}/data_n{n}_d{d}.npz"

            if not os.path.exists(filename):
                print(f"Skipping: {filename} not found.")
                continue

            data = np.load(filename)
            X, y = data['X'], data['y']

            start_time = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y)
            end_time = time.time()
            elapsed_time = end_time - start_time

            results.append({
                'Samples (n)': n,
                'Dimensions (d)': d,
                'Time (s)': round(elapsed_time, 4)
            })

    return pd.DataFrame(results)

def generate_and_save_datasets(d_list, n_list, u_val=100, folder="test_data"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for d in d_list:
        for n in n_list:
            print(f"Generating: n={n}, d={d}...")


            X, X_test, y, y_test = make_classification(n=n, d=d, u=u_val, random_state=1)
            X_full = np.vstack((X, X_test))
            y_full = np.hstack((y, y_test))

            filename = f"{folder}/data_n{n}_d{d}.npz"
            np.savez(filename, X=X_full, y=y_full)


if __name__ == "__main__":
    #Task 1
    X,X_test, Y, Y_test = make_classification(d=2, n=100, u=10,random_state=1)
    X_combined = np.vstack((X, X_test))
    y_combined = np.hstack((Y, Y_test))
    svc = LinearSVC(eta=0.00001, n_iter=1000, C=1.0)
    svc.fit(X, Y)
    plot_decision_regions(X_combined, y_combined,classifier=svc,test_idx=range(len(X),len(X_combined)))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    #Task 3 and 4
    d_scales = [10, 50, 100]
    n_scales = [100, 500, 5000]

    generate_and_save_datasets(d_scales, n_scales)
    #Task 3
    svc = LinearSVC(eta=0.00001, n_iter=1000, C=1.0)
    df_results = run_scalability_test(svc, d_scales, n_scales)


    pivot_table = df_results.pivot(index='Samples (n)', columns='Dimensions (d)', values='Time (s)')
    print("\n--- Time Cost (Seconds) ---")
    print(pivot_table)


    #Task 4
    run_sklearn_test(d_scales, n_scales)
    dual = sklearnSVC(dual=True)
    primal = sklearnSVC(dual=False)

    df_results_dual = run_sklearn_time_test(dual,d_scales, n_scales)
    df_results_primal = run_sklearn_time_test(primal,d_scales, n_scales)


    pivot_table_dual = df_results_dual.pivot(index='Samples (n)', columns='Dimensions (d)', values='Time (s)')
    print("\n--- Time Cost (Seconds) For Dual---")
    print(pivot_table_dual)

    pivot_table_primal = df_results_primal.pivot(index='Samples (n)', columns='Dimensions (d)', values='Time (s)')
    print("\n--- Time Cost (Seconds) For Primal---")
    print(pivot_table_primal)



