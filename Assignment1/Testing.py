import pandas as pd

from AdalineGD import AdalineGD
from LogisticRegressionGD import LogisticRegressionGD
import matplotlib.pyplot as plt
from MultiClassAdalineGD import *
from matplotlib.colors import ListedColormap
from AdalineSGD import *

def plot_data(X, Y,dataset_name):
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

# Task 2
X , Y = get_scaled_iris()
plot_data(X, Y,"Iris")

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

df_wine = df_wine[df_wine[0].isin([1,2])]
Y = df_wine.iloc[:, 0].values
Y = np.where(Y == 1, 0, 1)
X = df_wine.iloc[:, 1:].values
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std


plot_data(X, Y,"Wine")

#Task 1
X , Y = get_scaled_iris()

ada = AdalineGD(eta=0.01, n_iter=500)
ada.fit(X, Y)
plot_decision_regions(X, Y, classifier=ada)
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.title('Adaline - Iris')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

X , Y = get_scaled_iris()

lrg = LogisticRegressionGD(eta=0.01, n_iter=5000)
lrg.fit(X, Y)
plot_decision_regions(X, Y, classifier=lrg)
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.title('Logistic Regression - Iris')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

#Task 3
df_iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
Y = df_iris.iloc[:, 4].values
Y = np.where(Y == 'Iris-setosa', 0, np.where(Y == 'Iris-versicolor',1,2))
X = df_iris.iloc[:, [0, 2]].values
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std

mca = MultiClassPerceptron(0.01,500)
mca.fit(X, Y)
plot_decision_regions(X, Y, classifier=mca)
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.title('Multi-Class Adaline - Iris')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

#Task 4
X, Y = get_scaled_iris()
adaSGD = AdalineSGD(eta=0.01, n_iter=500)
ada = AdalineGD(eta=0.01, n_iter=500)
lrg = LogisticRegressionGD(eta=0.01, n_iter=500)

adaSGD.fit(X, Y)
ada.fit(X, Y)
lrg.fit_mini_batch_SGD(X, Y)

plot_decision_regions(X, Y, classifier=lrg)
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.title('Lrg Mini Batch - Iris')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
