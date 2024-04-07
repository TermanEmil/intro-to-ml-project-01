import dtuimldmtools
import numpy as np
import sklearn.model_selection
import sklearn.linear_model
from sklearn.preprocessing import LabelEncoder

from main import importData2
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def main():
    data = importData2().standardized()
    X = data.X
    y = data.classLabels

    attributeNames = data.attributeNames
    classNames = data.classNames

    N, M = X.shape
    C = len(classNames)

    lambda_interval = np.logspace(-8, 2, 50)
    K = 5
    CV = sklearn.model_selection.KFold(K, shuffle=True)
    for train_index, test_index in CV.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        model = LogisticRegression(
            solver="lbfgs", multi_class="multinomial", tol=1e-4, random_state=1, C=1 / lambda_interval[0]
        )
        model.fit(X_train, y_train)

        y_train_est = model.predict(X_train).T
        y_test_est = model.predict(X_test).T

        print(y_train_est)
        print(y_test_est)



if __name__ == '__main__':
    main()
