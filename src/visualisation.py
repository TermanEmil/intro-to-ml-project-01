import numpy as np
from matplotlib import pyplot as plt

from main import importData2


attributeCombos = [
    (0, 1), (0, 2), (2, 6),
    (2, 5), (2, 4), (0, 6),
]


def visualise2DAttributes():
    """
    Visualise different combinations of attributes on a 2D plot.
    """

    data = importData2()

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
        (0, 1), (0, 2), (1, 2),
    ]

    figure, axs = plt.subplots(1, 3)
    figure.suptitle('Grain properties: PCAs')
    figure.set_figheight(5)
    figure.set_figwidth(15)

    for i, ax in enumerate(axs.flat):
        pcaIndex1, pcaIndex2 = pcaIndexCombo[i]
        for classIndex in range(len(data.classNames)):
            classMask = data.classLabels == classIndex
            xData = Z[classMask, pcaIndex1]
            yData = Z[classMask, pcaIndex2]
            ax.plot(xData, yData, 'o', alpha=0.3)
        ax.set_xlabel(f'PC{pcaIndex1 + 1}')
        ax.set_ylabel(f'PC{pcaIndex2 + 1}')
        ax.legend(data.classNames)


def visualisePcaCoefficients():
    data = importData2().standardized()
    U, S, V, Z = data.computePca()

    figure = plt.figure(figsize=(8, 10))
    ax = figure.add_subplot()
    figure.suptitle("PC Coefficients")

    pcasCount = data.attributesCount
    bw = 0.2
    r = np.arange(1, data.attributesCount + 1)
    for i in range(pcasCount)[:3]:
        ax.bar(r + i * bw, V[:, i], width=bw)
    ax.set_xticks(r + bw, data.attributeNames)
    ax.set_xlabel("Attributes")
    ax.set_ylabel("Component coefficients")
    ax.legend([f'PC{i + 1}' for i in range(pcasCount)])
    ax.grid()


def visualisePcaInOriginalData():
    """
    Draw the PCA vectors on the original standardized data
    """

    data = importData2().standardized()
    U, S, V, Z = data.standardized().computePca()

    figure, axs = plt.subplots(2, 3)
    figure.suptitle('Grain properties with PC vectors')

    pcaColors = ['tab:pink', 'tab:olive', 'tab:cyan']
    pcaIndexes = [0, 1]
    pcaLabels = [f'PC{i + 1}' for i in pcaIndexes]
    lastPcaPlots = []

    for i, ax in enumerate(axs.flat):
        # Plot the data points
        attributeIndex1, attributeIndex2 = attributeCombos[i]
        for classIndex in range(len(data.classNames)):
            classMask = data.classLabels == classIndex
            xData = data.X[classMask, attributeIndex1]
            yData = data.X[classMask, attributeIndex2]
            ax.plot(xData, yData, 'o', alpha=0.3, label=data.classNames[classIndex])
        ax.set_xlabel(data.attributeNames[attributeIndex1])
        ax.set_ylabel(data.attributeNames[attributeIndex2])

        # Plot the PCA vectors on top
        lastPcaPlots = []
        for pcaIndex in pcaIndexes:
            # Extract a specific PCA
            fullPca = V[:, pcaIndex]

            # Extract the values specific to the chosen combinations of attributes
            pca = np.array([fullPca[attributeIndex1], fullPca[attributeIndex2]])

            # Set limits so that the plot is not scaled when huge lines are drawn
            # This is necessary when drawing the PCA lines
            x = data.X[:, attributeIndex1]
            y = data.X[:, attributeIndex2]
            ax.set_xlim(np.min(x) * 1.1, np.max(x) * 1.1)
            ax.set_ylim(np.min(y) * 1.1, np.max(y) * 1.1)

            # Plot a line across the PCA vector (using a hack by drawing 2 arrows)
            arrowSize = 2
            pcaPlot = ax.arrow(
                0, 0, pca[0] * 100, pca[1] * 100,
                head_width=0, head_length=0,
                color=pcaColors[pcaIndex],
                label=f'PC{pcaIndex + 1}'
            )
            lastPcaPlots.append(pcaPlot)
            ax.arrow(
                0, 0, -pca[0] * 100, -pca[1] * 100,
                head_width=0, head_length=0,
                color=pcaColors[pcaIndex],
            )
            # Plot the PCA vector with an arrow in the vector's direction
            ax.arrow(
                0, 0, pca[0], pca[1],
                head_width=arrowSize*0.05, head_length=arrowSize*0.03,
            )

    # Legend for the data points
    figure.legend(data.classNames)
    # Legend for PCA using the last drawn plots (this is a slight hack)
    figure.legend(lastPcaPlots, pcaLabels, loc='lower right')


def visualiseAllAttributes():
    data = importData2().standardized()

    figure = plt.figure(figsize=(12, 12))
    figure.suptitle('Grain properties')
    M = data.attributesCount

    for m1 in range(data.attributesCount):
        for m2 in range(data.attributesCount):
            ax = plt.subplot(M, M, m1 * M + m2 + 1)
            for classIndex in range(len(data.classNames)):
                classMask = data.classLabels == classIndex
                xData = data.X[classMask, m1]
                yData = data.X[classMask, m2]
                ax.plot(xData, yData, '.', alpha=0.4)
                if m1 == data.attributesCount - 1:
                    ax.set_xlabel(data.attributeNames[m2])
                else:
                    ax.set_xticks([])
                if m2 == 0:
                    ax.set_ylabel(data.attributeNames[m1])
                else:
                    ax.set_yticks([])
    figure.legend(data.classNames)


visualise2DAttributes()
visualisePca()
visualisePcaCoefficients()
visualisePcaInOriginalData()
visualiseAllAttributes()
plt.show()
