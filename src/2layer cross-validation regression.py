import numpy as np
import pandas as pd
from sklearn import model_selection
from dtuimldmtools import rlr_validate, train_neural_net
import torch
from regression_data_transformation import import_regression_data


X_r, y_r, attribute_names = import_regression_data('compactness')
N, M = X_r.shape

# Regularization strength and hidden units
rs = np.power(10.0, range(-7, 5))
hu = np.arange(1, 5)                    # change to right ones!!

# Define ANN model
h = hu[0]
ann_model = lambda: torch.nn.Sequential(
    torch.nn.Linear(M, h),
    torch.nn.Tanh(),
    torch.nn.Linear(h, 1)
)
loss_fn = torch.nn.MSELoss()
max_iter = 10000

# Add offset attribute for regularization
X_r = np.concatenate((np.ones((X_r.shape[0], 1)), X_r), 1)
attribute_names = ["Offset"] + attribute_names

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K, shuffle=True)

k1 = 0
E_test_reg = np.empty(K)
E_test_ann = np.empty(K)
E_test_bas = np.empty(K)
M_reg = np.empty(K)
M_ann = np.empty(K)

for par_index, test_index in CV.split(X_r, y_r):
    print("\nOuter crossvalidation fold: {0}/{1}".format(k1 + 1, K))

    # extract training and test set for current outer CV fold
    X_par, y_par = X_r[par_index, :], y_r[par_index]
    X_test, y_test = X_r[test_index, :], y_r[test_index]

    # Determine test error for baseline model
    E_test_bas[k1] = np.sum((y_test - np.mean(y_par))**2) / len(y_test)

    # Find optimal value for regularization strength
    (
        opt_val_err,
        M_reg[k1],
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_par, y_par, rs, K)

    # Train model on training data
    M = M + 1
    mu = np.mean(X_par[:, 1:], 0)
    sigma = np.std(X_par[:, 1:], 0)
    X_par_stand = X_par.copy()
    X_par_stand[:, 1:] = (X_par[:, 1:] - mu) / sigma
    X_test_stand = X_test.copy()
    X_test_stand[:, 1:] = (X_test[:, 1:] - mu) / sigma
    Xty = X_par_stand.T @ y_par
    XtX = X_par_stand.T @ X_par_stand
    lambdaI = M_reg[k1] * np.eye(M)
    lambdaI[0, 0] = 0
    w_rlr = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

    # Determine test error for optimal regularization strength
    E_test_reg[k1] = np.square(y_test - X_test_stand @ w_rlr).sum(axis=0) / len(y_test)

    # Remove offset for ANN
    X_par = X_par[:, 1:]
    X_test = X_test[:, 1:]
    M = M - 1

    k2 = 0
    E_val_ann = np.empty((len(hu), K))
    for train_index, val_index in CV.split(X_par, y_par):
        print("\nANN crossvalidation fold: {0}/{1}".format(k2 + 1, K))

        # Extract training and test set for current inner CV fold
        X_train = torch.Tensor(X_par[train_index, :])
        y_train = torch.Tensor(y_par[train_index])
        X_val = torch.Tensor(X_par[val_index, :])
        y_val = torch.Tensor(y_par[val_index])

        # Train ANN model with different hidden units
        for i in range(len(hu)):
            h = hu[i]
            print("Training model of type:\n{}\n".format(str(ann_model())))
            ann, final_loss, learning_curve = train_neural_net(
                ann_model, loss_fn, X=X_train, y=y_train, n_replicates=3, max_iter=max_iter
            )

            # Determine validation error
            y_est = ann(X_val)
            se = list()
            for j in range(len(y_est)):
                se.append((y_est[j] - y_val[j]) ** 2)
            E_val_ann[i, k2] = (sum(se).type(torch.float) / len(y_val)).data.numpy()
        
        k2 += 1
    
    # Choose best ANN model
        # right axis for summing across K not models? and how to deal with different lengths of y_train?
    E_gen_ann = np.sum(len(y_train)/len(y_par) * E_val_ann, axis=1)
    opt_hu_idx = np.argmin(E_gen_ann)
    M_ann[k1] = hu[opt_hu_idx]

    # Train ANN model on training data
    X_par = torch.Tensor(X_par)
    y_par = torch.Tensor(y_par)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    h = int(M_ann[k1])
    print("Training model of type:\n{}\n".format(str(ann_model())))
    ann, final_loss, learning_curve = train_neural_net(
        ann_model, loss_fn, X=X_par, y=y_par, n_replicates=3, max_iter=max_iter
    )

    # Determine test error for best number of hidden units
    y_est = ann(X_test)
    se = list()
    for j in range(len(y_est)):
        se.append((y_est[j] - y_test[j]) ** 2)
    E_test_ann[k1] = (sum(se).type(torch.float) / len(y_test)).data.numpy()
    k1 += 1

# Output as table
comparison = pd.DataFrame(np.column_stack([M_ann,E_test_ann,M_reg,E_test_reg,E_test_bas]),
                     index=np.arange(1, K+1),
                     columns=["h*","E_test ANN","l*","E_test reg","E_test base"])
print(comparison)