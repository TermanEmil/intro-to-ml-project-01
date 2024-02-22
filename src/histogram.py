from matplotlib.pyplot import hist, show, subplot, title, xlabel
import numpy as np
from main import importData2
from matplotlib import pyplot as plt


def visualise_histogram():
    """
    Visualising Histogram on attributes of the data
    refer exercise 4_3_1
    """

    data = importData2()
    M = data.attributesCount
    C = len(data.classNames)
    X = data.X
    y = data.classLabels
    N = data.observationsCount
    attributeName = data.attributeNames

    figure = plt.figure()
    figure.suptitle("Seed Histogram")
    u = np.floor(np.sqrt(M))
    v = np.ceil(float(M) / u)
    for i in range(M):
        subplot(int(u), int(v), i + 1)
        hist(X[:, i])
        xlabel(attributeName[i])
       
    show()


visualise_histogram()