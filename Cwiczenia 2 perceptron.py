import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib.colors import ListedColormap

class Perceptron(object):
    def __init__(self, eta=0.001, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

def num_of_unique_elements(arr):
    return len(np.unique(arr))

class MultiClass:
    def __init__(self, train_data, train_label):
        self.perceptrons = []
        counts = num_of_unique_elements(train_label)
        for i in range(counts):
            label_copy = train_label.copy()
            for j in range(len(label_copy)):
                if label_copy[j] != i:
                    label_copy[j] = -1
                else:
                    label_copy[j] = 1
            perceptron = Perceptron(eta=0.001, n_iter=100)
            perceptron.fit(train_data, label_copy)
            self.perceptrons.append(perceptron)

    def predict(self, X):
        prep_arr = []
        for i in range(len(X)):
            temp = []
            for perceptron in self.perceptrons:
                temp.append(perceptron.net_input(X[i]))
            class_index = np.argmax(temp)
            prep_arr.append(class_index)
        return np.array(prep_arr)

def plot_decision_regions(X, y, classifier):
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])  
    markers = ('s', 'x', 'o')  
    labels = ('Klasa 0', 'Klasa 1', 'Klasa 2') 

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.1))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap.colors[idx],
                    marker=markers[idx], label=labels[cl])

    plt.legend()
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    classifier = MultiClass(X_train, y_train)

    predictions = classifier.predict(X_test)

    plot_decision_regions(X=X_test, y=predictions, classifier=classifier)

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
