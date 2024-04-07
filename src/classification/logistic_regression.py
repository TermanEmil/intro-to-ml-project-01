import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from main import importData2


random_state = 1


def main():
    data = importData2().standardized()
    X = data.X
    y = data.classLabels

    # # Delete some attributes to see how the training behaves
    # X = np.delete(X, [True, True, True, False, False, False, False], 1)

    # Using ex8_1_2:
    # Create crossvalidation partition for evaluation using stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)

    # No need to standardize because we are standardized it at the import
    # --- omitted standardization part ---

    # Fit regularized logistic regression model to training data to predict the class label
    lambda_interval = np.logspace(-8, 8, 1000)

    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    for k in range(0, len(lambda_interval)):
        model = LogisticRegression(
            random_state=random_state,
            multi_class="multinomial", solver='lbfgs',
            # multi_class='ovr', solver='liblinear',
            # solver='newton-cg',
            # solver='saga',
            # solver='newton-cholesky',
            # solver='sag',
            max_iter=20000,
            penalty="l2",
            C=1 / lambda_interval[k]
        )

        model.fit(X_train, y_train)

        y_train_est = model.predict(X_train).T
        y_test_est = model.predict(X_test).T

        train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

        w_est = model.coef_[0]
        coefficient_norm[k] = np.sqrt(np.sum(w_est ** 2))

    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]

    plt.figure(figsize=(8, 8))
    plt.semilogx(lambda_interval, train_error_rate * 100)
    plt.semilogx(lambda_interval, test_error_rate * 100)
    plt.semilogx(opt_lambda, min_error * 100, "o")
    plt.text(
        1e-8,
        3,
        "Minimum test error: "
        + str(np.round(min_error * 100, 2))
        + " % at 1e"
        + str(np.round(np.log10(opt_lambda), 2)),
    )
    plt.xlabel("Regularization strength, $\log_{10}(\lambda)$")
    plt.ylabel("Error rate (%)")
    plt.title("Classification error")
    plt.legend(["Training error", "Test error", "Test minimum"], loc="upper right")
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.semilogx(lambda_interval, coefficient_norm, "k")
    plt.ylabel("L2 Norm")
    plt.xlabel("Regularization strength, $\log_{10}(\lambda)$")
    plt.title("Parameter vector L2 norm")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
