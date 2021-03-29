""" Fonction nécessaire au bon fonctionnement du tp3 en ML """
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.utils import resample
from sklearn import metrics
from sklearn.metrics import classification_report
from time import *

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    """
    Fonction découverte sur le net dans le but d'afficher les classificateurs obtenus.

    Référence: https://chih-ling-hsu.github.io/2017/08/30/NN-XOR
    """

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')

def split_xy(df):
    xTrain = df.iloc[:, 1:]
    yTrain = df.iloc[:, 0]
    return xTrain, yTrain

# Fonction qui effectue un echantillonnage une validation croisée
def ResampleAndScore(clf, Trainset, n_samples, cv):
    """La fonction fait la cross-validation et appelle l'autre fonction pour print les résultats."""
    sample = resample(Trainset, n_samples=n_samples, replace=False, stratify=Trainset['revenue'], random_state=0)
    xTrain, yTrain = split_xy(sample)
    print("test")
    print("--Pour",n_samples, "échantillons avec le dataset",Trainset.name, "--")
    y_pred = cross_val_predict(clf, xTrain, yTrain, cv=cv)
    print(classification_report(yTrain, y_pred))

    del sample, xTrain, yTrain, y_pred
