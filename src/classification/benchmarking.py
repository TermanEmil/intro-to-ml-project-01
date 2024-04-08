import os

import numpy as np
import progressbar
import torch
import random

from dtuimldmtools import train_neural_net
from sklearn import model_selection
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from main import importData2


random_state = 1


def benchmark_logistic_regression_model():
    random.seed(random_state)
    np.random.seed(random_state)

    outer_folds_count = 10
    inner_folds_count = 5

    max_iterations = 20000

    data = importData2().standardized()
    X = data.X
    y = data.classLabels

    lambda_values = np.logspace(-5, 2, 3000)

    lambda_value_with_its_test_error_rate = np.empty(outer_folds_count, dtype=tuple)

    outer_folder = model_selection.StratifiedKFold(n_splits=outer_folds_count, shuffle=True, random_state=random_state)
    for outer_fold, (outer_train_i, outer_test_i) in enumerate(outer_folder.split(X, y)):
        outer_X_train, outer_y_train = X[outer_train_i, :], y[outer_train_i]
        outer_X_test, outer_y_test = X[outer_test_i, :], y[outer_test_i]

        print(
            f'\n{"=" * 20} '
            f'Outer crossvalidation fold: {outer_fold + 1}/{outer_folds_count} '
            f'(train: {len(outer_y_train)} | test: {len(outer_y_test)})'
        )

        # Compute the testing error of different lambda_values across multiples folds
        inner_test_error_rate = np.empty((len(lambda_values), inner_folds_count))

        inner_folder = model_selection.KFold(n_splits=inner_folds_count, shuffle=True, random_state=random_state)
        for inner_fold, (inner_train_i, inner_test_i) in enumerate(inner_folder.split(outer_X_train, outer_y_train)):
            inner_X_train, inner_y_train = outer_X_train[inner_train_i, :], outer_y_train[inner_train_i]
            inner_X_test, inner_y_test = outer_X_train[inner_test_i, :], outer_y_train[inner_test_i]
            print(
                f'{"-" * 10} Inner crossvalidation fold: {inner_fold + 1}/{inner_folds_count} '
                f'(train: {len(inner_y_train)} | test: {len(inner_y_test)})'
            )

            bar = progressbar.ProgressBar(
                maxval=len(lambda_values),
                widgets=['Loading: ', progressbar.Bar('*')]
            ).start()

            # Train the model using different lambda_values
            for i in range(len(lambda_values)):
                model = LogisticRegression(
                    random_state=random_state,
                    multi_class="multinomial", solver='lbfgs',
                    max_iter=max_iterations,
                    penalty="l2",
                    C=1 / lambda_values[i]
                )

                model.fit(inner_X_train, inner_y_train)
                test_predictions = model.predict(inner_X_test).T
                inner_test_error_rate[i, inner_fold] = np.sum(test_predictions != inner_y_test) / len(inner_y_test)
                bar.update(i)
            print()

        # Choose the best (optimal) lambda_value
        optimal_lambda_value_index = np.argmin(inner_test_error_rate.mean(axis=1))
        optimal_lambda_value = lambda_values[optimal_lambda_value_index]

        optimal_model = LogisticRegression(
            random_state=random_state,
            multi_class="multinomial", solver='lbfgs',
            max_iter=max_iterations,
            penalty="l2",
            C=1 / optimal_lambda_value
        )
        optimal_model.fit(outer_X_train, outer_y_train)
        optimal_test_predictions = optimal_model.predict(outer_X_test).T
        error_rate = np.sum(optimal_test_predictions != outer_y_test) / len(outer_y_test)
        lambda_value_with_its_test_error_rate[outer_fold] = (optimal_lambda_value, error_rate)

    print(lambda_value_with_its_test_error_rate)
    return lambda_value_with_its_test_error_rate


def benchmark_ann_model():
    random.seed(random_state)
    np.random.seed(random_state)

    outer_folds_count = 10
    inner_folds_count = 5

    max_iterations = 3000
    n_replicates = 1

    # Use the general cross entropy loss for multinomial classification
    loss_fn = torch.nn.CrossEntropyLoss()

    data = importData2().standardized()
    X = data.X
    y = data.classLabels
    N, M = X.shape
    C = len(data.classNames)

    # The complexity parameter is the number of neurons in the only hidden layer
    hidden_units_range = list(range(1, 18 + 1))

    # Define the ANN model
    def create_ann_model(hidden_neurons_count: int):
        return lambda: torch.nn.Sequential(
            torch.nn.Linear(M, hidden_neurons_count),
            torch.nn.ReLU(),

            torch.nn.Linear(hidden_neurons_count, C),
            torch.nn.Softmax(dim=1),
        )

    complexity_param_with_its_test_error_rate = np.empty(outer_folds_count, dtype=tuple)

    outer_folder = model_selection.StratifiedKFold(n_splits=outer_folds_count, shuffle=True, random_state=random_state)
    for outer_fold, (outer_train_i, outer_test_i) in enumerate(outer_folder.split(X, y)):
        outer_X_train, outer_y_train = X[outer_train_i, :], y[outer_train_i]
        outer_X_test, outer_y_test = X[outer_test_i, :], y[outer_test_i]

        print(
            f'\n{"=" * 20} '
            f'Outer crossvalidation fold: {outer_fold + 1}/{outer_folds_count} '
            f'(train: {len(outer_y_train)} | test: {len(outer_y_test)})'
        )

        # Compute the testing error of different complexity parameters across multiples folds
        inner_test_error_rate = np.empty((len(hidden_units_range), inner_folds_count))

        inner_folder = model_selection.KFold(n_splits=inner_folds_count, shuffle=True, random_state=random_state)
        for inner_fold, (inner_train_i, inner_test_i) in enumerate(inner_folder.split(outer_X_train, outer_y_train)):
            inner_X_train, inner_y_train = outer_X_train[inner_train_i, :], outer_y_train[inner_train_i]
            inner_X_test, inner_y_test = outer_X_train[inner_test_i, :], outer_y_train[inner_test_i]
            print(
                f'{"-" * 10} Inner crossvalidation fold: {inner_fold + 1}/{inner_folds_count} '
                f'(train: {len(inner_y_train)} | test: {len(inner_y_test)})'
            )

            # Train the model using the range of complexity parameters
            for i in range(len(hidden_units_range)):
                net, final_loss, learning_curve = train_neural_net(
                    create_ann_model(hidden_units_range[i]),
                    loss_fn,
                    X=torch.tensor(inner_X_train, dtype=torch.float),
                    y=torch.tensor(inner_y_train, dtype=torch.long),
                    n_replicates=n_replicates,
                    max_iter=max_iterations
                )
                print("\n\tBest loss: {}\n".format(final_loss))

                # Determine probability of each class using trained network
                softmax_logits = net(torch.tensor(inner_X_test, dtype=torch.float))
                test_predictions = (torch.max(softmax_logits, dim=1)[1]).data.numpy()
                inner_test_error_rate[i, inner_fold] = np.sum(test_predictions != inner_y_test) / len(inner_y_test)
            print()

        # Choose the optimal complexity parameter
        optimal_complexity_param_index = np.argmin(inner_test_error_rate.mean(axis=1))
        optimal_complexity_param = hidden_units_range[optimal_complexity_param_index]

        optimal_net, optimal_final_loss, _ = train_neural_net(
            create_ann_model(optimal_complexity_param),
            loss_fn,
            X=torch.tensor(outer_X_train, dtype=torch.float),
            y=torch.tensor(outer_y_train, dtype=torch.long),
            n_replicates=n_replicates,
            max_iter=max_iterations
        )

        # Determine probability of each class using trained network
        softmax_logits = optimal_net(torch.tensor(outer_X_test, dtype=torch.float))
        optimal_test_predictions = (torch.max(softmax_logits, dim=1)[1]).data.numpy()

        error_rate = np.sum(optimal_test_predictions != outer_y_test) / len(outer_y_test)
        complexity_param_with_its_test_error_rate[outer_fold] = (optimal_complexity_param, error_rate)

    print(complexity_param_with_its_test_error_rate)
    return complexity_param_with_its_test_error_rate


def benchmark_baseline_model():
    random.seed(random_state)
    np.random.seed(random_state)

    outer_folds_count = 10

    data = importData2().standardized()
    X = data.X
    y = data.classLabels

    test_error_rate = np.empty(outer_folds_count)

    outer_folder = model_selection.StratifiedKFold(n_splits=outer_folds_count, shuffle=True, random_state=random_state)
    for outer_fold, (outer_train_i, outer_test_i) in enumerate(outer_folder.split(X, y)):
        outer_X_train, outer_y_train = X[outer_train_i, :], y[outer_train_i]
        outer_X_test, outer_y_test = X[outer_test_i, :], y[outer_test_i]

        model = DummyClassifier(strategy='most_frequent')
        model.fit(outer_X_train, outer_y_train)
        test_predictions = model.predict(outer_X_test)
        test_error_rate[outer_fold] = np.sum(test_predictions != outer_y_test) / len(outer_y_test)

    print(test_error_rate)
    print(f'Average error rate across all folds: {np.mean(test_error_rate)}')
    return test_error_rate


def main():
    # benchmark_logistic_regression_model()
    # benchmark_ann_model()
    benchmark_baseline_model()
    pass


if __name__ == '__main__':
    main()
