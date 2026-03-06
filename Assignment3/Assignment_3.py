from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


def digit_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    mnist_images, mnist_labels  = mnist.data, mnist.target
    return mnist_images[:60000], mnist_images[60000:], mnist_labels[:60000], mnist_labels[60000:]

def fashion_data():
    mnist_fashion = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
    mnist_fashion_images, mnist_fashion_labels  = mnist_fashion.data, mnist_fashion.target
    return mnist_fashion_images[:60000],mnist_fashion_images[60000:], mnist_fashion_labels[:60000], mnist_fashion_labels[60000:]

X_train,X_test,y_train,y_test = digit_data()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



