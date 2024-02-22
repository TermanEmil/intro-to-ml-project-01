import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import hist, show, subplot, xlabel

from main import importData2


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
        xlabel(attributeName[i])

        # Stacked histograms
        classMasks = [data.classLabels == classIndex for classIndex in range(len(data.classNames))]
        histData = [X[mask, i] for mask in classMasks]
        hist(histData, alpha=0.7, label=data.classNames, stacked=True)
        # plt.legend(data.classNames, bot)

    figure.legend(data.classNames,  bbox_to_anchor=(0.9, 0.2))
    show()


visualise_histogram()