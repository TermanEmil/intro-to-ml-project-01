from matplotlib import pyplot

from main import importData2


def visualise2DAttributes():
    data = importData2()

    attributeCombos = [
        (0, 1), (0, 2), (2, 6),
        (0, 4), (0, 5), (2, 4),
    ]

    figure, axs = pyplot.subplots(2, 3)
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

    pyplot.show()


visualise2DAttributes()
