import numpy

from src.main import importData2


def import_regression_data(label_to_predict: str) -> [numpy.ndarray, numpy.ndarray]:
    data = importData2()
    non_standardized_X = data.X
    standardized_X = data.standardized().X

    prediction_index = data.attributeNames.index(label_to_predict)

    Y_r = non_standardized_X[:, prediction_index]

    # Delete the column that we want to predict
    X_r = numpy.delete(standardized_X, prediction_index, axis=1)

    X_r_attribute_names = data.attributeNames.copy()
    X_r_attribute_names.remove(label_to_predict)
    return X_r, Y_r, X_r_attribute_names


def sandbox():
    """Proof of concept of how to import the data"""
    label_to_predict = "compactness"
    X_r, Y_r, X_r_attribute_names = import_regression_data(label_to_predict)
    print(X_r)
    print(Y_r)
    print(X_r_attribute_names)


if __name__ == "__main__":
    sandbox()

