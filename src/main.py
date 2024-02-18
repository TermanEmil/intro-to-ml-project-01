import dataclasses
from dataclasses import dataclass
from functools import cached_property
from typing import List, Tuple
from scipy import linalg

import numpy as np
import pandas as pd


def importRawData() -> np.ndarray:
    filename = '../data/seeds_dataset.txt'
    # There are rows with multiple tabs, we have to use a regex delimiter separating by one or more separators.
    df = pd.read_csv(filename, delimiter=r"\s+")
    return df.values


def transformSpeciesColumnIntoOneOutOfKEncoding(data: np.ndarray) -> np.ndarray:
    """
    There are 3 different species
    The original data has as its last column encoded the species as a value from 1 to 3.
    The following code replaces that column with one-out-of-K encoding with 3 binary columns
    """
    species = np.array(data[:, -1], dtype=int).T
    K = species.max()
    speciesEncoding = np.zeros((species.size, K))
    speciesEncoding[np.arange(species.size), species - 1] = 1
    return np.concatenate((data[:, :-1], speciesEncoding), axis=1)


def importData() -> Tuple[np.ndarray, List]:
    rawData = importRawData()
    X = transformSpeciesColumnIntoOneOutOfKEncoding(rawData)
    attributeNames = [
        'Area A',
        'perimeter P',
        'compactness C',
        'length of kernel',
        'width of kernel',
        'asymmetry coefficient',
        'length of kernel groove',
        'Kama', 'Rosa', 'Canadian'
    ]
    return X, attributeNames


@dataclass
class MlData:
    classNames: List[str]
    attributeNames: List[str]
    X: np.ndarray
    classLabels: np.ndarray

    @property
    def observationsCount(self):
        return len(self.classLabels)

    @property
    def attributesCount(self):
        return self.X.shape[1]

    @property
    def dataFrame(self):
        return pd.DataFrame(self.X, columns=self.attributeNames)

    def centered(self) -> 'MlData':
        # Subtract the mean from the data
        data = dataclasses.replace(self)
        data.X = data.X - np.ones((data.observationsCount, 1)) * data.X.mean(axis=0)
        return data

    def standardized(self) -> 'MlData':
        # Center and divide by the attribute standard deviation to obtain a standardized dataset
        data = self.centered()
        data.X = data.X * (1 / np.std(data.X, axis=0))
        return data

    # noinspection PyTupleAssignmentBalance
    def computePca(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: (
            U matrix from svd();
            S matrix from svd();
            V matrix from svd() but transposed;
            Z matrix with the data projected onto PCA space;
        )
        """

        # PCA by computing SVD of Y
        U, S, Vh = linalg.svd(self.X, full_matrices=False)

        # scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
        # of the vector V. So, for us to obtain the correct V, we transpose:
        V = Vh.T

        # Project the centered data onto principal component space
        Z = self.X @ V
        return U, S, V, Z


def importData2() -> MlData:
    rawData = importRawData()
    # Delete the row containing the species
    X = np.delete(rawData, -1, axis=1)

    classNames = ['Kama', 'Rosa', 'Canadian']
    attributeNames = [
        'Area A',
        'perimeter P',
        'compactness C',
        'length of kernel',
        'width of kernel',
        'asymmetry coefficient',
        'length of kernel groove',
    ]
    classLabels = rawData[:, -1] - 1
    return MlData(classNames=classNames, attributeNames=attributeNames, X=X, classLabels=classLabels)
