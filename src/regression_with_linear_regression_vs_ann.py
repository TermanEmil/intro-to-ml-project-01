import random

import numpy as np
import torch
from dtuimldmtools import rlr_validate, train_neural_net
from dtuimldmtools.statistics.statistics import correlated_ttest
from sklearn import model_selection

from main import importData2

random_state = 1
random.seed(random_state)
np.random.seed(random_state)
torch.random.manual_seed(random_state)

ann_max_iterations = 3000
ann_n_replicates = 3

outer_folds_count = 10
inner_folds_count = 5


def import_regression_data(label_to_predict: str) -> [np.ndarray, np.ndarray]:
    data = importData2()
    X = data.standardized().X

    prediction_index = data.attributeNames.index(label_to_predict)

    Y_r = X[:, prediction_index]

    # Delete the column that we want to predict
    X_r = np.delete(X, prediction_index, axis=1)

    X_r_attribute_names = data.attributeNames.copy()
    X_r_attribute_names.remove(label_to_predict)
    return X_r, Y_r, X_r_attribute_names


def _build_outer_fold_header_str(fold_i, total_folds, y_train, y_test):
    return (
        f'\n{"=" * 20} '
        f'Outer crossvalidation fold: {fold_i + 1}/{total_folds} '
        f'(train: {len(y_train)} | test: {len(y_test)})'
    )


def _build_inner_fold_header_str(fold_i, total_folds, y_train, y_test):
    return (
        f'{"-" * 10} '
        f'Inner crossvalidation fold: {fold_i + 1}/{total_folds} '
        f'(train: {len(y_train)} | test: {len(y_test)})'
    )


def _create_ann_model(M: int, hidden_neurons_count: int):
    return lambda: torch.nn.Sequential(
        torch.nn.Linear(M, hidden_neurons_count),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_neurons_count, 1),
    )


def benchmark_ann():
    random.seed(random_state)
    np.random.seed(random_state)
    torch.random.manual_seed(random_state)

    X, y, attribute_names = import_regression_data('compactness')
    N, M = X.shape

    hidden_units_range = list(range(1, 24 + 1))
    loss_fn = torch.nn.MSELoss()

    complexity_param_with_its_test_error_rate = np.empty(outer_folds_count, dtype=tuple)

    outer_folder = model_selection.KFold(n_splits=outer_folds_count, shuffle=True, random_state=random_state)
    for outer_fold, (outer_train_i, outer_test_i) in enumerate(outer_folder.split(X, y)):
        outer_X_train, outer_y_train = X[outer_train_i, :], y[outer_train_i]
        outer_X_test, outer_y_test = X[outer_test_i, :], y[outer_test_i]

        print(_build_outer_fold_header_str(outer_fold, outer_folds_count, outer_y_train, outer_y_test))

        inner_test_error_rate = np.empty((len(hidden_units_range), inner_folds_count))

        inner_folder = model_selection.KFold(n_splits=inner_folds_count, shuffle=True, random_state=random_state)
        for inner_fold, (inner_train_i, inner_test_i) in enumerate(inner_folder.split(outer_X_train, outer_y_train)):
            inner_X_train, inner_y_train = outer_X_train[inner_train_i, :], outer_y_train[inner_train_i]
            inner_X_test, inner_y_test = outer_X_train[inner_test_i, :], outer_y_train[inner_test_i]
            print(_build_inner_fold_header_str(inner_fold, inner_folds_count, inner_y_train, inner_y_test))

            # Train ANN model with different hidden units
            for i in range(len(hidden_units_range)):
                ann, _, _ = train_neural_net(
                    _create_ann_model(M, hidden_units_range[i]),
                    loss_fn,
                    X=torch.tensor(inner_X_train, dtype=torch.float),
                    y=torch.tensor([[x] for x in inner_y_train], dtype=torch.float),
                    n_replicates=ann_n_replicates,
                    max_iter=ann_max_iterations
                )

                # Determine validation error
                predictions = ann(torch.tensor(inner_X_test, dtype=torch.float))
                inner_test_error_rate[i, inner_fold] = np.mean([
                    (predictions[j].detach().numpy() - inner_y_test[j]) ** 2
                    for j in range(len(predictions))
                ])

        # Find the optimal complexity parameter based on the error rate in the inner fold
        optimal_complexity_param_index = np.argmin(inner_test_error_rate.mean(axis=1))
        optimal_complexity_param = hidden_units_range[optimal_complexity_param_index]

        # Train ANN model using optimal complexity parameter
        optimal_ann, _, _ = train_neural_net(
            _create_ann_model(M, optimal_complexity_param),
            loss_fn,
            X=torch.tensor(outer_X_train, dtype=torch.float),
            y=torch.tensor([[x] for x in outer_y_train], dtype=torch.float),
            n_replicates=ann_n_replicates,
            max_iter=ann_max_iterations
        )

        # Determine test error for best number of hidden units
        optimal_predictions = optimal_ann(torch.tensor(outer_X_test, dtype=torch.float))
        error_rate = np.mean([
            (optimal_predictions[j].detach().numpy() - outer_y_test[j]) ** 2
            for j in range(len(outer_y_test))
        ])
        print('#' * 5, f'Error rate with optimal param: {error_rate} (h = {optimal_complexity_param}')
        complexity_param_with_its_test_error_rate[outer_fold] = (optimal_complexity_param, error_rate)

    print(complexity_param_with_its_test_error_rate)
    return complexity_param_with_its_test_error_rate


def benchmark_linear_regression():
    random.seed(random_state)
    np.random.seed(random_state)
    torch.random.manual_seed(random_state)

    X, y, _ = import_regression_data('compactness')
    N, M = X.shape
    M = M + 1

    # Add offset attribute for regularization
    X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)

    lambdas = np.power(10.0, range(-7, 5))
    complexity_param_with_its_test_error_rate = np.empty(outer_folds_count, dtype=tuple)

    outer_folder = model_selection.KFold(n_splits=outer_folds_count, shuffle=True, random_state=random_state)
    for outer_fold, (outer_train_i, outer_test_i) in enumerate(outer_folder.split(X, y)):
        outer_X_train, outer_y_train = X[outer_train_i, :], y[outer_train_i]
        outer_X_test, outer_y_test = X[outer_test_i, :], y[outer_test_i]

        print(_build_outer_fold_header_str(outer_fold, outer_folds_count, outer_y_train, outer_y_test))

        # Find optimal value for regularization strength
        (
            opt_val_err,
            opt_lambda,
            mean_w_vs_lambda,
            train_err_vs_lambda,
            test_err_vs_lambda,
        ) = rlr_validate(outer_X_train, outer_y_train, lambdas, inner_folds_count)

        # Train model on training data
        mu = np.mean(outer_X_train[:, 1:], 0)
        sigma = np.std(outer_X_train[:, 1:], 0)
        X_par_stand = outer_X_train.copy()
        X_par_stand[:, 1:] = (outer_X_train[:, 1:] - mu) / sigma
        X_test_stand = outer_X_test.copy()
        X_test_stand[:, 1:] = (outer_X_test[:, 1:] - mu) / sigma
        Xty = X_par_stand.T @ outer_y_train
        XtX = X_par_stand.T @ X_par_stand
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0, 0] = 0
        w_rlr = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

        # Determine test error for optimal regularization strength
        error_rate = np.square(outer_y_test - X_test_stand @ w_rlr).sum(axis=0) / len(outer_y_test)
        print('#' * 5, f'Error rate with optimal param: {error_rate}')

        complexity_param_with_its_test_error_rate[outer_fold] = (opt_lambda, error_rate)

    print(complexity_param_with_its_test_error_rate)
    return complexity_param_with_its_test_error_rate


def benchmark_baseline():
    random.seed(random_state)
    np.random.seed(random_state)
    torch.random.manual_seed(random_state)

    X, y, attribute_names = import_regression_data('compactness')

    test_error_rates = np.empty(outer_folds_count)

    outer_folder = model_selection.KFold(n_splits=outer_folds_count, shuffle=True, random_state=random_state)
    for outer_fold, (outer_train_i, outer_test_i) in enumerate(outer_folder.split(X, y)):
        outer_X_train, outer_y_train = X[outer_train_i, :], y[outer_train_i]
        outer_X_test, outer_y_test = X[outer_test_i, :], y[outer_test_i]

        # Determine test error for baseline model
        predictions = np.mean(outer_y_train)
        test_error_rates[outer_fold] = np.mean((outer_y_test - predictions) ** 2)

    print(test_error_rates)
    return test_error_rates


def statistical_evaluation():
    random.seed(random_state)
    np.random.seed(random_state)
    torch.random.manual_seed(random_state)

    X, y, _ = import_regression_data('compactness')
    N, M = X.shape

    X_linear_reg = np.copy(X)

    # Add offset attribute for regularization
    X_linear_reg = np.concatenate((np.ones((X.shape[0], 1)), X), 1)

    optimal_lambda = 0.001
    optimal_hidden_layer_size = 23
    loss_fn = torch.nn.MSELoss()

    runs_count = 1

    linear_regression_errors = []
    ann_errors = []
    baseline_errors = []

    for run_i in range(runs_count):
        random_seed = random_state + run_i

        print('\n', '=' * 20, f'Run {run_i + 1}/{runs_count}', '=' * 20)

        folder = model_selection.KFold(n_splits=outer_folds_count, shuffle=True, random_state=random_seed)
        for fold, (train_i, test_i) in enumerate(folder.split(X, y)):
            # Linear Regression
            X_lr_train, y_train = X_linear_reg[train_i, :], y[train_i]
            X_lr_test, y_test = X_linear_reg[test_i, :], y[test_i]
            mu = np.mean(X_lr_train[:, 1:], 0)
            sigma = np.std(X_lr_train[:, 1:], 0)
            X_par_stand = X_lr_train.copy()
            X_par_stand[:, 1:] = (X_lr_train[:, 1:] - mu) / sigma
            X_test_stand = X_lr_test.copy()
            X_test_stand[:, 1:] = (X_lr_test[:, 1:] - mu) / sigma
            Xty = X_par_stand.T @ y_train
            XtX = X_par_stand.T @ X_par_stand
            lambdaI = optimal_lambda * np.eye(M + 1)
            lambdaI[0, 0] = 0
            w_rlr = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
            lr_error = np.square(y_test - X_test_stand @ w_rlr).sum(axis=0) / len(y_test)
            linear_regression_errors.append(lr_error)

            # ANN
            X_train, y_train = X[train_i, :], y[train_i]
            X_test, y_test = X[test_i, :], y[test_i]
            ann, _, _ = train_neural_net(
                _create_ann_model(M, optimal_hidden_layer_size),
                loss_fn,
                X=torch.tensor(X_train, dtype=torch.float),
                y=torch.tensor([[x] for x in y_train], dtype=torch.float),
                n_replicates=ann_n_replicates,
                max_iter=ann_max_iterations
            )
            ann_predictions = ann(torch.tensor(X_test, dtype=torch.float))
            ann_error = np.mean([
                (ann_predictions[j].detach().numpy() - y_test[j]) ** 2
                for j in range(len(y_test))
            ])
            ann_errors.append(ann_error)

            # Baseline
            predictions = np.mean(y_train)
            baseline_error = np.mean((y_test - predictions) ** 2)
            baseline_errors.append(baseline_error)

    folds_count = len(baseline_errors)
    r_ann_bas = np.empty(folds_count)
    r_ann_reg = np.empty(folds_count)
    r_reg_bas = np.empty(folds_count)

    for fold in range(folds_count):
        E_ann = ann_errors[fold]
        E_bas = baseline_errors[fold]
        E_reg = linear_regression_errors[fold]

        # Determine difference in error for use in setup II
        r_ann_bas[fold] = np.mean(E_ann - E_bas)
        r_ann_reg[fold] = np.mean(E_ann - E_reg)
        r_reg_bas[fold] = np.mean(E_reg - E_bas)

    # Initialize values for setup II
    alpha = 0.05
    rho = 1/folds_count

    # Find pairwise p-values and confidence intervals
    p_ann_bas, CI_ann_bas = correlated_ttest(r_ann_bas, rho, alpha=alpha)
    p_ann_reg, CI_ann_reg = correlated_ttest(r_ann_reg, rho, alpha=alpha)
    p_reg_bas, CI_reg_bas = correlated_ttest(r_reg_bas, rho, alpha=alpha)

    print(f'Statistical evaluation')
    print(f'ANN vs Linear reg: p-value: {p_ann_reg}; CI: {CI_ann_reg}')
    print(f'ANN vs baseline: p-value: {p_ann_bas}; CI: {CI_ann_bas}')
    print(f'Linear reg vs baseline: p-value: {p_reg_bas}; CI: {CI_reg_bas}')


if __name__ == '__main__':
    # benchmark_ann()
    # benchmark_linear_regression()
    # benchmark_baseline()
    statistical_evaluation()
