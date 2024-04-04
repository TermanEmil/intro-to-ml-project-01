import numpy as np
import pandas as pd
from sklearn import model_selection
from dtuimldmtools import rlr_validate, train_neural_net
import torch
from regression_data_transformation import import_regression_data
from statistics import multimode
from dtuimldmtools.statistics.statistics import correlated_ttest


X_r, y_r, attribute_names = import_regression_data('compactness')
N, M = X_r.shape

# Regularization strength and hidden units
rs = np.power(10.0, range(-7, 5))
hu = np.arange(1, 7)

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

# Save splits for use in statistical comparison
splits = []
for par_index, test_index in CV.split(X_r, y_r):
    splits.append((par_index,test_index))
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

# Convert to numpy arrays for multimode
M_ann = np.array(M_ann)
M_reg = np.array(M_reg)
E_test_ann = np.array(E_test_ann)
E_test_reg = np.array(E_test_reg)

# Find most common optimal number of hidden units
opt_hu = multimode(M_ann)
# If multiple are equally common find the one of them with lowest errors
if len(opt_hu)>1:
    err = []
    for i in range(len(opt_hu)):
        err.append(sum(E_test_ann[M_ann==opt_hu[i]]))
    opt_hu_ind = np.argmin(err)
    opt_hu = opt_hu[opt_hu_ind]
else:
    opt_hu = opt_hu[0]
print("Optimal hidden units:",opt_hu)

# Find most common optimal regularization stength
opt_rs = multimode(M_reg)
# If multiple are equally common find the one of them with lowest errors
if len(opt_rs)>1:
    err = []
    for i in range(len(opt_rs)):
        err.append(sum(E_test_reg[M_reg==opt_rs[i]]))
    opt_rs_ind = np.argmin(err)
    opt_rs = opt_rs[opt_rs_ind]
else:
    opt_rs = opt_rs[0]
print("Optimal regulation strenght:",opt_rs)

# Run outer crossvalidation loop again with the same folds
r_ann_bas = np.empty(K)
r_ann_reg = np.empty(K)
r_reg_bas = np.empty(K)
k=0
for par_index, test_index in splits:
    X_par, y_par = X_r[par_index, :], y_r[par_index]
    X_test, y_test = X_r[test_index, :], y_r[test_index]

    # Determine test error for baseline model
    E_bas = (y_test - np.mean(y_par))**2
    
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
    lambdaI = opt_rs * np.eye(M)
    lambdaI[0, 0] = 0
    w_rlr = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

    # Determine test error for optimal regularization strength
    E_reg = np.square(y_test - X_test_stand @ w_rlr)
    
    # Remove offset for ANN
    X_par = X_par[:, 1:]
    X_test = X_test[:, 1:]
    M = M - 1
    
    # Train ANN model on training data
    X_par = torch.Tensor(X_par)
    y_par = torch.Tensor(y_par)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    h = int(opt_hu)
    ann, final_loss, learning_curve = train_neural_net(
        ann_model, loss_fn, X=X_par, y=y_par, n_replicates=3, max_iter=max_iter
    )

    # Determine test error for best number of hidden units
    y_est = ann(X_test)
    se = np.empty(len(y_est))
    for j in range(len(y_est)):
        se[j] = (y_est[j] - y_test[j]) ** 2
    E_ann = se

    # Determine difference in error for use in setup II
    r_ann_bas[k] = np.mean(E_ann - E_bas)
    r_ann_reg[k] = np.mean(E_ann - E_reg)
    r_reg_bas[k] = np.mean(E_reg - E_bas)

    k += 1

# Initialize values for setup II
alpha = 0.05
rho = 1/K

# Find pairwise p-values and confidence intervals
p_ann_bas, CI_ann_bas = correlated_ttest(r_ann_bas, rho, alpha=alpha)
p_ann_reg, CI_ann_reg = correlated_ttest(r_ann_reg, rho, alpha=alpha)
p_reg_bas, CI_reg_bas = correlated_ttest(r_reg_bas, rho, alpha=alpha)
print("ANN vs base. p:", p_ann_bas, "CI:", CI_ann_bas)
print("Regularised vs base. p:", p_reg_bas, "CI:", CI_reg_bas)
print("ANN vs regularised. p:", p_ann_reg, "CI:", CI_ann_reg)
