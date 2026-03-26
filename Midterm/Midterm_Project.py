import time
from collections import Counter

import idx2numpy
import numpy as np
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
    train_img_path = 'mnist/train-images-idx3-ubyte'
    train_lbl_path = 'mnist/train-labels-idx1-ubyte'
    test_img_path = 'mnist/t10k-images-idx3-ubyte'
    test_lbl_path = 'mnist/t10k-labels-idx1-ubyte'

    X_train = idx2numpy.convert_from_file(train_img_path)
    y_train = idx2numpy.convert_from_file(train_lbl_path)
    X_test = idx2numpy.convert_from_file(test_img_path)
    y_test = idx2numpy.convert_from_file(test_lbl_path)

    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    return X_train, X_test, y_train, y_test

def fashion_data():
    train_img_path = 'fashion/train-images-idx3-ubyte'
    train_lbl_path = 'fashion/train-labels-idx1-ubyte'
    test_img_path = 'fashion/t10k-images-idx3-ubyte'
    test_lbl_path = 'fashion/t10k-labels-idx1-ubyte'

    X_train = idx2numpy.convert_from_file(train_img_path)
    y_train = idx2numpy.convert_from_file(train_lbl_path)
    X_test = idx2numpy.convert_from_file(test_img_path)
    y_test = idx2numpy.convert_from_file(test_lbl_path)

    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    return X_train, X_test, y_train, y_test



def pca_lda_compare(data_name):
    if data_name == 'mnist_784':
        X_train, X_test, y_train, y_test = digit_data()
    else:
        X_train, X_test, y_train, y_test = fashion_data()

    reductions = {
        'PCA_50': PCA(n_components=50,random_state=1),
        'PCA_100': PCA(n_components=100,random_state=1),
        'PCA_200': PCA(n_components=200,random_state=1),
        'LDA': LDA()
    }

    pca_lda_results = []

    for name, reducer in reductions.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            (name, reducer),
            ('svc', SVC(kernel='rbf',random_state=1))
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
        'linear' : SVC(kernel='linear',C=1,cache_size=2000,random_state=1),
        'rbf' : SVC(kernel='rbf',C=10,gamma=0.001,cache_size=2000,random_state=1),
        'poly' : SVC(kernel='poly',C=1,gamma=0.01,degree=3,cache_size=2000,random_state=1),
    }

    pca_vals = [50,100,200]

    for name, kernel in kernels.items():
        results = []
        for pca_val in pca_vals:

            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('reducer', PCA(n_components=pca_val,random_state=1)),
                ('svc', kernel),
            ])

            X_train_transforemed = pipe[:-1].fit_transform(X_train)
            X_test_transforemed = pipe[:-1].transform(X_test)

            start_time = time.time()
            pipe = pipe[-1]
            pipe.fit(X_train_transforemed, y_train)
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


def run_task_4_bagging(data_name):
    if data_name == 'mnist_784':
        X_train, X_test, y_train, y_test = digit_data()
    else:
        X_train, X_test, y_train, y_test = fashion_data()

    kernels = {
        'linear': SVC(kernel='linear', C=1, cache_size=2000,random_state=1),
        'rbf': SVC(kernel='rbf', C=10, gamma=0.001, cache_size=2000,random_state=1),
        'poly': SVC(kernel='poly', C=1, gamma=0.01, degree=3, cache_size=2000,random_state=1),
    }

    scaler = StandardScaler()
    pca = PCA(n_components=100,random_state=1)

    X_train_comp = pca.fit_transform(scaler.fit_transform(X_train))
    X_test_comp = pca.transform(scaler.transform(X_test))

    n_models = 8
    subset_size = len(X_train_comp) // n_models
    results = []

    for name, base_svc in kernels.items():
        start_single = time.time()
        single_model = base_svc
        single_model.fit(X_train_comp, y_train)
        single_time = time.time() - start_single
        single_preds = single_model.predict(X_test_comp)
        single_error = 1 - accuracy_score(y_test, single_preds)


        bagging_start = time.time()
        all_models = []

        for i in range(n_models):
            start_idx = i * subset_size
            end_idx = (i + 1) * subset_size
            X_sub, y_sub = X_train_comp[start_idx:end_idx], y_train[start_idx:end_idx]

            model = SVC(**base_svc.get_params())
            model.fit(X_sub, y_sub)
            all_models.append(model)


        bagging_time = time.time() - bagging_start

        all_subset_preds = []

        for model in all_models:
            all_subset_preds.append(model.predict(X_test_comp))


        all_subset_preds = np.array(all_subset_preds)
        final_preds = []
        for j in range(len(X_test_comp)):
            vote = Counter(all_subset_preds[:, j]).most_common(1)[0][0]
            final_preds.append(vote)

        bagging_error = 1 - accuracy_score(y_test, final_preds)
        results.append({
            'Kernel': name,
            'Single Time (s)': round(single_time, 2),
            'Single Error': round(single_error, 4),
            'Bagging Time (s)': round(bagging_time, 2),
            'Bagging Error': round(bagging_error, 4)
        })

    df_results = pd.DataFrame(results)
    print(f"\n--- Table for {data_name}: Bootstrap Aggregating ---")
    print(df_results.to_string(index=False))



if __name__ == '__main__':
    # pca_lda_compare('mnist_784')
    # pca_lda_compare('Fashion-MNIST')
    # print("Test 2 \n\n\n")
    compare_kernels('mnist_784')
    compare_kernels('Fashion-MNIST')
    # print("Test 3 \n\n\n")
    # run_task_4_bagging('mnist_784')
    # run_task_4_bagging('Fashion-MNIST')