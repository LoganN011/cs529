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
    if data_name == 'mnist_784':
        X_train, X_test, y_train, y_test = digit_data()
    else:
        X_train, X_test, y_train, y_test = fashion_data()
    kernels = {
        'linear' : SVC(kernel='linear',C=1,cache_size=2000), #hit its limit
        'rbf' : SVC(kernel='rbf',C=10,gamma=0.01,cache_size=2000), #should lower C to help with overfitting or increase gamma
        'poly' : SVC(kernel='poly',C=1,gamma=0.01,degree=3,cache_size=2000), #The best with low training error and low test error
    }

    pca_vals = [50,100,200]

    for name, kernel in kernels.items():
        results = []
        for pca_val in pca_vals:

            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('reducer', PCA(n_components=pca_val)),
                ('svc', kernel),
            ])

            X_train_transforemed = pipe[:-1].fit_transform(X_train)
            X_test_transforemed = pipe[:-1].transform(X_test)

            start_time = time.time()
            pipe = pipe[-1].fit(X_train_transforemed, y_train)
            total_train_time = time.time() - start_time


            error_test = pipe.score(X_test_transforemed, y_test)
            error_train = pipe.score(X_train_transforemed, y_train)

            results.append({
            'Method': name,
            'PCA': pca_val,
            'Total Train Time (s)': round(total_train_time, 2),
            'Test Error':  round(1 - error_test, 4),
            'Training Error': round(1 - error_train, 4)
        })



        df_kernal = pd.DataFrame(results)

        print(f"\n--- Table for {data_name}: Three kernels in SVC ---")
        print(df_kernal.to_string(index=False))






# pca_lad_compare('mnist_784')
# pca_lad_compare('Fashion-MNIST')

compare_kernels('mnist_784')
print('test2')
compare_kernels('Fashion-MNIST')