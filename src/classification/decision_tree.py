import importlib_resources
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pylab import figure, legend, plot, show, xlabel, ylabel
from scipy.io import loadmat
from sklearn import model_selection, tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from main import importData2


def main():
    # exercise 6.1.1
    data = importData2().standardized()
    X = data.X
    y = np.array(data.classLabels, dtype=int).T
    # y = data.oneOutOfKEncodedClassLabels

    # Tree complexity parameter - constraint on maximum depth
    tc = np.arange(2, 10, 1)

    # Simple holdout-set crossvalidation
    # Create crossvalidation partition for evaluation using stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    uniqueTrain, countsTrain = np.unique(y_train, return_counts=True)
    print(f'Classes count in train data {dict(zip(uniqueTrain, countsTrain))}')

    uniqueTest, countsTest = np.unique(y_test, return_counts=True)
    print(f'Classes count in test data {dict(zip(uniqueTest, countsTest))}')

    # Initialize variables
    Error_train = np.empty((len(tc), 1))
    Error_test = np.empty((len(tc), 1))

    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion="gini", max_depth=t)
        dtc = dtc.fit(X_train, y_train)

        # Evaluate classifier's misclassification rate over train/test data
        y_est_test = np.asarray(dtc.predict(X_test), dtype=int)
        y_est_train = np.asarray(dtc.predict(X_train), dtype=int)
        misclass_rate_test = sum(y_est_test != y_test) / float(len(y_est_test))
        misclass_rate_train = sum(y_est_train != y_train) / float(len(y_est_train))
        Error_test[i], Error_train[i] = misclass_rate_test, misclass_rate_train

        # Confusion matrix
        # cm = confusion_matrix(y_test, y_est_test)
        # cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.classNames)
        # cm_display.plot()
        # plt.show()

    f = figure()
    plot(tc, Error_train * 100)
    plot(tc, Error_test * 100)
    xlabel("Model complexity (max tree depth)")
    ylabel("Error (%)")
    legend(["Error_train", "Error_test"])
    show()


if __name__ == "__main__":
    main()
