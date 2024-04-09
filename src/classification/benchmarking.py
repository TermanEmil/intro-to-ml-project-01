import random

import numpy as np
import pandas as pd
import progressbar
import torch
from dtuimldmtools import train_neural_net, mcnemar, correlated_ttest
from sklearn import model_selection
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate

from main import importData2

random_state = 1

# Logistic regression constants
logistic_regression_max_iterations = 20000

# ANN constants
ann_max_iterations = 3000
ann_n_replicates = 3


def _build_outer_fold_header_str(fold_i, total_folds, y_train, y_test):
    return (
        f'\n{"=" * 20} '
        f'Outer crossvalidation fold: {fold_i + 1}/{total_folds} '
        f'(train: {len(y_train)} | test: {len(y_test)})'
        f'\n\tTrain class distribution: {sum(y_train == 0)} | {sum(y_train == 1)} | {sum(y_train == 2)}'
        f'\n\tTest class distribution: {sum(y_test == 0)} | {sum(y_test == 1)} | {sum(y_test == 2)}'
    )


def benchmark_logistic_regression_model():
    random.seed(random_state)
    np.random.seed(random_state)
    torch.random.manual_seed(random_state)

    outer_folds_count = 10
    inner_folds_count = 5

    data = importData2().standardized()
    X = data.X
    y = data.classLabels

    lambda_values = np.logspace(-5, 2, 3000)

    lambda_value_with_its_test_error_rate = np.empty(outer_folds_count, dtype=tuple)

    outer_folder = model_selection.KFold(n_splits=outer_folds_count, shuffle=True, random_state=random_state)
    for outer_fold, (outer_train_i, outer_test_i) in enumerate(outer_folder.split(X, y)):
        outer_X_train, outer_y_train = X[outer_train_i, :], y[outer_train_i]
        outer_X_test, outer_y_test = X[outer_test_i, :], y[outer_test_i]

        print(_build_outer_fold_header_str(outer_fold, outer_folds_count, outer_y_train, outer_y_test))

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
                    max_iter=logistic_regression_max_iterations,
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
            max_iter=logistic_regression_max_iterations,
            penalty="l2",
            C=1 / optimal_lambda_value
        )
        optimal_model.fit(outer_X_train, outer_y_train)
        optimal_test_predictions = optimal_model.predict(outer_X_test).T
        error_rate = np.sum(optimal_test_predictions != outer_y_test) / len(outer_y_test)
        lambda_value_with_its_test_error_rate[outer_fold] = (optimal_lambda_value, error_rate)

    print(lambda_value_with_its_test_error_rate)
    return lambda_value_with_its_test_error_rate


def _create_ann_model(M: int, C: int, hidden_neurons_count: int):
    return lambda: torch.nn.Sequential(
        torch.nn.Linear(M, hidden_neurons_count),
        torch.nn.ReLU(),

        torch.nn.Linear(hidden_neurons_count, C),
        torch.nn.Softmax(dim=1),
    )


def benchmark_ann_model():
    random.seed(random_state)
    np.random.seed(random_state)
    torch.random.manual_seed(random_state)

    outer_folds_count = 10
    inner_folds_count = 5

    # Use the general cross entropy loss for multinomial classification
    loss_fn = torch.nn.CrossEntropyLoss()

    data = importData2().standardized()
    X = data.X
    y = data.classLabels
    N, M = X.shape
    C = len(data.classNames)

    # The complexity parameter is the number of neurons in the only hidden layer
    hidden_units_range = list(range(1, 18 + 1))

    complexity_param_with_its_test_error_rate = np.empty(outer_folds_count, dtype=tuple)

    outer_folder = model_selection.KFold(n_splits=outer_folds_count, shuffle=True, random_state=random_state)
    for outer_fold, (outer_train_i, outer_test_i) in enumerate(outer_folder.split(X, y)):
        outer_X_train, outer_y_train = X[outer_train_i, :], y[outer_train_i]
        outer_X_test, outer_y_test = X[outer_test_i, :], y[outer_test_i]

        print(_build_outer_fold_header_str(outer_fold, outer_folds_count, outer_y_train, outer_y_test))

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
                    _create_ann_model(M, C, hidden_units_range[i]),
                    loss_fn,
                    X=torch.tensor(inner_X_train, dtype=torch.float),
                    y=torch.tensor(inner_y_train, dtype=torch.long),
                    n_replicates=ann_n_replicates,
                    max_iter=ann_max_iterations
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
            _create_ann_model(M, C, optimal_complexity_param),
            loss_fn,
            X=torch.tensor(outer_X_train, dtype=torch.float),
            y=torch.tensor(outer_y_train, dtype=torch.long),
            n_replicates=ann_n_replicates,
            max_iter=ann_max_iterations
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
    torch.random.manual_seed(random_state)

    outer_folds_count = 10

    data = importData2().standardized()
    X = data.X
    y = data.classLabels

    test_error_rate = np.empty(outer_folds_count)

    outer_folder = model_selection.KFold(n_splits=outer_folds_count, shuffle=True, random_state=random_state)
    for outer_fold, (outer_train_i, outer_test_i) in enumerate(outer_folder.split(X, y)):
        outer_X_train, outer_y_train = X[outer_train_i, :], y[outer_train_i]
        outer_X_test, outer_y_test = X[outer_test_i, :], y[outer_test_i]

        model = DummyClassifier(strategy='most_frequent')
        # model = DummyClassifier(strategy='stratified')
        model.fit(outer_X_train, outer_y_train)
        test_predictions = model.predict(outer_X_test)
        test_error_rate[outer_fold] = np.sum(test_predictions != outer_y_test) / len(outer_y_test)

    print(test_error_rate)
    print(f'Average error rate across all folds: {np.mean(test_error_rate)}')
    return test_error_rate


def benchmark_pairwise_comparison():
    random.seed(random_state)
    np.random.seed(random_state)
    torch.random.manual_seed(random_state)

    folds_count = 10
    runs_count = 1000

    data = importData2().standardized()
    X = data.X
    y = data.classLabels
    N, M = X.shape
    C = len(data.classNames)

    # # Delete a class
    # X = np.delete(X, y == 2, axis=0)
    # y = np.delete(y, y == 2)

    # Using the lambda from the second fold that gave a test error of 0
    logistic_regression_optimal_lambda = 0.005992285967408437

    # Using 10 hidden units according to the ANN benchmark results
    ann_optimal_hidden_units = 10

    y_true = []
    all_logistic_regression_predictions = []
    all_ann_predictions = []
    all_baseline_predictions = []

    logistic_regression_coefficients = []

    try:
        for run_i in range(runs_count):
            print('\n', '=' * 20, f'Run {run_i + 1}/{runs_count}', '=' * 20)
            random_seed = random_state + run_i
            folder = model_selection.KFold(n_splits=folds_count, shuffle=True, random_state=random_seed)
            for fold_i, (train_i, test_i) in enumerate(folder.split(X, y)):
                print('-' * 10, f'fold: {fold_i + 1}/{folds_count}', '-' * 10)

                X_train, y_train = X[train_i, :], y[train_i]
                X_test, y_test = X[test_i, :], y[test_i]

                y_true.append(y_test)

                # Logistic Regression
                logistic_regression_model = LogisticRegression(
                    random_state=random_seed,
                    multi_class="multinomial", solver='lbfgs',
                    max_iter=logistic_regression_max_iterations,
                    penalty="l2",
                    C=1 / logistic_regression_optimal_lambda
                )
                logistic_regression_model.fit(X_train, y_train)
                logistic_regression_predictions = logistic_regression_model.predict(X_test).T
                all_logistic_regression_predictions.append(logistic_regression_predictions)
                logistic_regression_coefficients.append(logistic_regression_model.coef_)

                # ANN
                ann_net, _, _ = train_neural_net(
                    _create_ann_model(M, C, ann_optimal_hidden_units),
                    torch.nn.CrossEntropyLoss(),
                    X=torch.tensor(X_train, dtype=torch.float),
                    y=torch.tensor(y_train, dtype=torch.long),
                    n_replicates=ann_n_replicates,
                    max_iter=ann_max_iterations
                )
                ann_predictions = (torch.max(ann_net(torch.tensor(X_test, dtype=torch.float)), dim=1)[1]).data.numpy()
                all_ann_predictions.append(ann_predictions)

                # Baseline
                baseline_model = DummyClassifier(strategy='most_frequent')
                baseline_model.fit(X_train, y_train)
                baseline_predictions = baseline_model.predict(X_test)
                all_baseline_predictions.append(baseline_predictions)
    except KeyboardInterrupt:
        min_size = min(
            len(y_true),
            len(all_logistic_regression_predictions),
            len(all_ann_predictions),
            len(all_baseline_predictions)
        )
        y_true = y_true[:min_size]
        all_logistic_regression_predictions = all_logistic_regression_predictions[:min_size]
        all_ann_predictions = all_ann_predictions[:min_size]
        all_baseline_predictions = all_baseline_predictions[:min_size]

    print(f'A total of {len(np.concatenate(y_true))} predictions')

    print('Coefficients for the first logistic regression')
    print(tabulate(pd.DataFrame(
        zip(data.attributeNames, np.around(np.transpose(logistic_regression_coefficients[0]), decimals=4)),
        columns=['features', 'coefficients']
    ), headers='keys', tablefmt='psql'))

    print('Mean of the absolute values of the coefficients for logistic regression')
    mean_of_absolute_values = np.around(np.mean(np.fabs(logistic_regression_coefficients), axis=0), decimals=4)
    print(tabulate(pd.DataFrame(
        zip(data.attributeNames, np.transpose(mean_of_absolute_values)),
        columns=['features', 'coefficients']
    ), headers='keys', tablefmt='psql'))

    print('Mean of the coefficients for logistic regression')
    mean_of_absolute_values = np.around(np.mean(logistic_regression_coefficients, axis=0), decimals=4)
    print(tabulate(pd.DataFrame(
        zip(data.attributeNames, np.transpose(mean_of_absolute_values)),
        columns=['features', 'coefficients']
    ), headers='keys', tablefmt='psql'))

    combinations = [
        ('Logistic vs ANN', all_logistic_regression_predictions, all_ann_predictions),
        ('Logistic vs Baseline', all_logistic_regression_predictions, all_baseline_predictions),
        ('ANN vs Baseline', all_ann_predictions, all_baseline_predictions)
    ]
    for title, model_a_predictions, model_b_predictions in combinations:
        print('\n', '=' * 20, f'{title}')
        print('McNemar')
        alpha = 0.05
        [theta, CI, p] = mcnemar(
            np.concatenate(y_true),
            np.concatenate(model_a_predictions),
            np.concatenate(model_b_predictions),
            alpha=alpha
        )
        print(f'theta = {theta}; CI: {CI}; p-value: {p}')

        print('\nSetup II')
        loss = 2
        r = [
            np.mean(np.abs(model_a_predictions[i] - y_true[i]) ** loss - np.abs(model_b_predictions[i] - y_true[i]))
            for i in range(len(y_true))
        ]
        rho = 1 / folds_count
        alpha = 0.05
        p_setupII, CI_setupII = correlated_ttest(r, rho, alpha=alpha)
        print(f'p_setupII = {p_setupII}; CI_setupII = {CI_setupII}')


def main():
    # benchmark_logistic_regression_model()
    # benchmark_ann_model()
    # benchmark_baseline_model()
    benchmark_pairwise_comparison()
    pass


if __name__ == '__main__':
    main()
