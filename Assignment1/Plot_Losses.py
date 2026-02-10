import pandas as pd
import numpy as np
from AdalineGD import AdalineGD
from LogisticRegressionGD import LogisticRegressionGD
import matplotlib.pyplot as plt

df_iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

Y = df_iris.iloc[:100, 4].values
Y = np.where(Y == 'Iris-setosa', 0, 1 )
X = df_iris.iloc[0:100, [0, 2]].values




def plot_data(X, Y,dataset_name):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    eta = .01
    n_iter = 500
    ada = AdalineGD(eta=eta, n_iter=n_iter)
    lrg = LogisticRegressionGD(eta=eta, n_iter=n_iter)
    ada.fit(X, Y)
    lrg.fit(X, Y)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    fig.suptitle(f'Dataset: {dataset_name}', fontsize=16, fontweight='bold')

    ax[0].plot(range(1, len(ada.losses_) + 1), ada.losses_, marker='o', color='blue')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Sum-squared-error')
    ax[0].set_title('Adaline - Learning rate '+ '%g' % eta)


    ax[1].plot(range(1, len(lrg.losses_) + 1), lrg.losses_, marker='o', color='orange')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Log-Loss')
    ax[1].set_title('Logistic Regression - Learning rate '+ '%g' % eta)

    plt.tight_layout()
    plt.show()

plot_data(X, Y,"Iris")

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

df_wine = df_wine[df_wine[0].isin([1,2])]
Y = df_wine.iloc[:, 0].values
Y = np.where(Y == 1, 0, 1)
X = df_wine.iloc[:, 1:].values


plot_data(X, Y,"Wine")