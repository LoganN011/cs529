import time

from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def digit_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    mnist_images, mnist_labels  = mnist.data, mnist.target
    return mnist_images[:60000], mnist_images[60000:], mnist_labels[:60000], mnist_labels[60000:]

def fashion_data():
    mnist_fashion = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
    mnist_fashion_images, mnist_fashion_labels  = mnist_fashion.data, mnist_fashion.target
    return mnist_fashion_images[:60000],mnist_fashion_images[60000:], mnist_fashion_labels[:60000], mnist_fashion_labels[60000:]

X_train,X_test,y_train,y_test = digit_data()


def pca_lad_compare(data_name):
    if data_name == 'mnist_784':
        X_train, X_test, y_train, y_test = digit_data()
    else:
        X_train, X_test, y_train, y_test = fashion_data()

    reductions = {
        'PCA_50': PCA(n_components=50),
        'PCA_100': PCA(n_components=100),
        'PCA_200': PCA(n_components=200),
        'LDA': LDA()
    }

    pca_lda_results = []

    for name, reducer in reductions.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            (name, reducer),
            ('svc', SVC(kernel='rbf'))
        ])

        start_time = time.time()
        pipe = pipe.fit(X_train, y_train)
        total_train_time = time.time() - start_time

        preds = pipe.predict(X_test)
        error = accuracy_score(y_test, preds)

        pca_lda_results.append({
            'Method': name,
            'Total Train Time (s)': round(total_train_time, 2),
            'Test Error': 1 - round(error, 4)
        })

    df_pca_lda = pd.DataFrame(pca_lda_results)

    print(f"\n--- Table for {data_name}: PCA vs LDA ---")
    print(df_pca_lda.to_string(index=False))


def compare_kernels(data_name):
    kernels = {
        'linear' : SVC(kernel='linear',C=1),
        'rbf' : SVC(kernel='rbf',C=1,gamma=0.1),
        'poly' : SVC(kernel='poly',C=1,gamma=0.1,degree=2),
    }

    pca_vals = [50,100,200]

    for name, kernel in kernels.items():
        for pca_val in pca_vals:

            if data_name == 'mnist_784':
                X_train, X_test, y_train, y_test = digit_data()
            else:
                X_train, X_test, y_train, y_test = fashion_data()

            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('reducer', PCA(n_components=pca_val)),
                ('svc', kernel),
            ])

            X_train = pipe.transform(X_train)
            X_test = pipe.transform(X_test)

            start_time = time.time()
            pipe = pipe.fit(X_train, y_train)
            total_train_time = time.time() - start_time

            preds = pipe.predict(X_test)
            error = accuracy_score(y_test, preds)

            print(f'name{name}, pca_val{pca_val}')
            print(total_train_time)
            print(error)






# pca_lad_compare('mnist_784')
# pca_lad_compare('Fashion-MNIST')

compare_kernels('mnist_784')
