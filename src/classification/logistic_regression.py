import dtuimldmtools
import numpy as np
import sklearn.model_selection
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from main import importData2
import matplotlib.pyplot as plt


def main():
    data = importData2().standardized()
    X = data.X
    y = np.array(data.classLabels, dtype=int).T

    # Using ex8_1_2:
    # Create crossvalidation partition for evaluation using stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)

    # No need to standardize because we are standardized it at the import
    # --- omitted standardization part ---

    # Fit regularized logistic regression model to training data to predict the class label
    lambda_interval = np.logspace(-8, 2, 100)
    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    for k in range(0, len(lambda_interval)):
        # mdl = LogisticRegression(penalty="l2", C=1 / lambda_interval[k])
        model = LogisticRegression(
            multi_class="multinomial",
            # solver='newton-cg',
            # solver='saga',
            # solver='newton-cholesky',
            # solver='sag',
            solver='lbfgs',
            max_iter=10000,
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
    # plt.plot(np.log10(lambda_interval), train_error_rate*100)
    # plt.plot(np.log10(lambda_interval), test_error_rate*100)
    # plt.plot(np.log10(opt_lambda), min_error*100, 'o')
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
    # plt.ylim([0, 4])
    plt.grid()
    plt.show()

    # plt.figure(figsize=(8, 8))
    # plt.semilogx(lambda_interval, coefficient_norm, "k")
    # plt.ylabel("L2 Norm")
    # plt.xlabel("Regularization strength, $\log_{10}(\lambda)$")
    # plt.title("Parameter vector L2 norm")
    # plt.grid()
    # plt.show()


if __name__ == '__main__':
    main()
