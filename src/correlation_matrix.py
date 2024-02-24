import numpy as np

from main import importData2, MlData
from matplotlib import pyplot as plt


def drawCorrelationMatrix(data: MlData):
    """
    Following a stackoverflow answer to Plot correlation matrix using pandas:
    https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
    """
    df = data.dataFrame
    correlationMatrix = df.corr()
    print(correlationMatrix)

    numericalData = df.select_dtypes(['number'])
    attributeNames = numericalData.columns

    # Plot the correlation matrix
    f = plt.figure()
    plt.matshow(correlationMatrix, fignum=f.number)
    plt.xticks(range(numericalData.shape[1]), attributeNames, rotation=-90, fontsize=7)
    plt.yticks(range(numericalData.shape[1]), attributeNames)
    plt.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)

    # Show the values directly on the grid
    for (i, j), z in np.ndenumerate(correlationMatrix):
        if z <= 0.1:
            color = 'white'
        else:
            color = 'black'
        plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', color=color)

    plt.colorbar()
    plt.title('Correlation Matrix')

    plt.show()


if __name__ == '__main__':
    drawCorrelationMatrix(importData2())
