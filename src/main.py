import dataclasses
from dataclasses import dataclass
from functools import cached_property
from typing import List, Tuple

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

    @cached_property
    def observationsCount(self):
        return len(self.classLabels)

    def centered(self):
        data = dataclasses.replace(self)
        data.X = data.X - np.ones((data.observationsCount, 1)) * data.X.mean(axis=0)
        return data


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
    ] + classNames
    classLabels = rawData[:, -1] - 1
    return MlData(classNames=classNames, attributeNames=attributeNames, X=X, classLabels=classLabels)
