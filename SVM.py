'''
Script to understand internal parameters of SVM with 2 features

Input  : Size of dataset to experiment
Output : Plot containing Generated data, Support vectors, separating hyperplane, 

High level flow:
1) Generates data (can be modified to generate interesting data)
2) Splits data into train, test datasets
3) Fits Linear SVM
4) Plots data and classifier, prints relevant internal variables

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_classification
import sys
import pandas as pd
from tabulate import tabulate
import random
import os

# TO DO:
# 1) Need to check why 0 <= \alpha <= C is violated

def create_data(size, params):
    u, C = params
    return np.random.multivariate_normal(u, C, size=size)

def create_all_classes_data(n_samples, my_data = True):
    if my_data:
        cluster_1 = ((5,5), np.eye(2))
        cluster_2 = ((-5,-5), np.eye(2))
        X1 = create_data(n_samples//2, cluster_1)
        X2 = create_data(n_samples//2, cluster_2)
        X = np.vstack((X1, X2))
        y = np.hstack((np.ones(n_samples//2), -1*np.ones(n_samples//2)))
    else:
        X, y = make_classification(n_samples=n_samples, n_informative=2, n_redundant=0, n_features=2, n_classes=2, 
                                n_clusters_per_class=1, class_sep=2.5, flip_y=0)
    return X, y

def plot_data(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, edgecolors='k')
    plt.grid(); plt.legend(); plt.xlabel('X1'); plt.ylabel('X2')
    plt.title('All Data')
    plt.show()

def plot_hyperplanes(X, y, clf):
    from sklearn.inspection import DecisionBoundaryDisplay
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    DecisionBoundaryDisplay.from_estimator(clf, X, plot_method="contour", colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--",  "-", "--"], ax=ax)
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors="none", edgecolors="k")
    a, b = clf.coef_[0]
    c = clf.intercept_[0]
    plt.title(f'Hyp eqn : ${a:.2f}x1 {b:+.2f}x2 {c:+.2f} = 0$')
    plt.grid(); plt.xlabel('x1'); plt.ylabel('x2')
    # plt.legend(); 
    plt.show()

if __name__ == '__main__':
    # seed=1234
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)

    size = 1000

    X, y = create_all_classes_data(size, my_data=False)
    # plot_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print(f"\nSupport vectors                  : \n{clf.support_vectors_}")
    a, b = clf.coef_[0]
    c = clf.intercept_[0]
    hyp_eqn = lambda x: np.dot(clf.coef_[0], x) + clf.intercept_[0]
    Xi_eqn = lambda x, y: 1-y*hyp_eqn(x)
    Margin = lambda x: np.abs(hyp_eqn(x)/np.linalg.norm(clf.coef_[0]))  # Considering perpendicular distance
    # for x in clf.support_vectors_:
    #     print(f"Margin({x[0]:+.2f}, {x[1]:+.2f})   : {np.abs(hyp_eqn(x)/np.linalg.norm(clf.coef_[0])):.2f}")
    textbook_y = y_train
    textbook_y[textbook_y==0] = -1  # Using format as in textbook
    df = pd.DataFrame({'Support Vector':[f"({x[0]:+.2f}, {x[1]:+.2f})" for x in clf.support_vectors_], 
                  'Margin': [Margin(x) for x in clf.support_vectors_],
                  'Alpha': clf.dual_coef_[0],
                  'Xi': [Xi_eqn(x, textbook_y[idx]) for x, idx in zip(clf.support_vectors_, clf.support_)],
                  })
    df['On support hyperplane?'] = 0
    df['On support hyperplane?'] = df['Xi'] < 0.01
    print(f'Separting Hyperplane equation       : {a:.2f}x1 {b:+.2f}x2 {c:+.2f} = 0')
    print()
    print(f"Final Parameters after optimization : ")
    print(tabulate(df, headers='keys', tablefmt='psql'))
    print("\nConfusion Matrix: ")
    print(confusion_matrix(y_test,y_pred))

    plot_hyperplanes(X_train, y_train, clf)
    print()