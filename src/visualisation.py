import numpy as np
from matplotlib import pyplot as plt

from main import importData2


def visualise2DAttributes():
    """
    Visualise different combinations of attributes on a 2D plot.
    """

    data = importData2()

    attributeCombos = [
        (0, 1), (0, 2), (2, 6),
        (0, 4), (0, 5), (2, 4),
    ]

    figure, axs = plt.subplots(2, 3)
    figure.suptitle('Grain properties')
    for i, ax in enumerate(axs.flat):
        attributeIndex1, attributeIndex2 = attributeCombos[i]
        for classIndex in range(len(data.classNames)):
            classMask = data.classLabels == classIndex
            xData = data.X[classMask, attributeIndex1]
            yData = data.X[classMask, attributeIndex2]
            ax.plot(xData, yData, 'o', alpha=0.3)
        ax.set_xlabel(data.attributeNames[attributeIndex1])
        ax.set_ylabel(data.attributeNames[attributeIndex2])
        ax.legend(data.classNames)


# noinspection PyTupleAssignmentBalance
def visualisePca():
    """
    Visualise different combinations for PCAs on a 2D plot.
    """

    data = importData2().standardized()
    U, S, V, Z = data.computePca()

    pcaIndexCombo = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3),
    ]

    figure, axs = plt.subplots(2, 3)
    figure.suptitle('Grain properties: PCAs')
    for i, ax in enumerate(axs.flat):
        pcaIndex1, pcaIndex2 = pcaIndexCombo[i]
        for classIndex in range(len(data.classNames)):
            classMask = data.classLabels == classIndex
            xData = Z[classMask, pcaIndex1]
            yData = Z[classMask, pcaIndex2]
            ax.plot(xData, yData, 'o', alpha=0.3)
        ax.set_xlabel(f'PCA{pcaIndex1}')
        ax.set_ylabel(f'PCA{pcaIndex2}')
        ax.legend(data.classNames)


def visualisePcaCoefficients():
    data = importData2().standardized()
    U, S, V, Z = data.computePca()

    pcasCount = data.attributesCount
    bw = 0.1
    r = np.arange(1, data.attributesCount + 1)
    for i in range(pcasCount):
        plt.bar(r + i * bw, V[:, i], width=bw)
    plt.xticks(r + bw, data.attributeNames)
    plt.xlabel("Attributes")
    plt.ylabel("Component coefficients")
    plt.legend([f'PC{i + 1}' for i in range(pcasCount)])
    plt.grid()
    plt.title("PCA Component Coefficients")


visualise2DAttributes()
visualisePca()
visualisePcaCoefficients()
plt.show()
