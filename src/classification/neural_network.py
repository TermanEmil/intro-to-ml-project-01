import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
import torch
from dtuimldmtools import train_neural_net, visualize_decision_boundary, draw_neural_net
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder

from main import importData2


def main():
    data = importData2().standardized()
    X = data.X
    y = data.classLabels

    N, M = X.shape
    C = len(data.classNames)

    attributeNames = data.attributeNames
    classNames = data.classNames

    max_iter = 10000

    n_hidden_units = 10
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),
        torch.nn.ReLU(),
        torch.nn.Linear(n_hidden_units, C),
        torch.nn.Softmax(dim=1),
    )

    summaries, summaries_axes = plt.subplots(1, 2, figsize=(10, 5))
    color_list = [
        "tab:orange",
        "tab:green",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "tab:red",
        "tab:blue",
    ]

    errors = []

    K = 3
    CV = model_selection.KFold(K, shuffle=True)
    for k, (train_index, test_index) in enumerate(CV.split(X, y)):
        print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]

        # use the general cross entropy loss for multinomial classification
        loss_fn = torch.nn.CrossEntropyLoss()

        net, final_loss, learning_curve = train_neural_net(
            model,
            loss_fn,
            X=torch.tensor(X_train, dtype=torch.float),
            y=torch.tensor(y_train, dtype=torch.long),
            n_replicates=3,
            max_iter=max_iter
        )
        print("\n\tBest loss: {}\n".format(final_loss))

        # Determine probability of each class using trained network
        softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
        y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy()

        # Determine errors
        e = y_test_est != y_test

        error_rate = sum(e) / len(y_test)
        errors.append(np.mean(error_rate))

        print(
            "Number of miss-classifications for ANN:\n\t {0} out of {1}. Error rate: {2}".format(
                sum(e),
                len(e),
                error_rate
            )
        )

        (h,) = summaries_axes[0].plot(learning_curve, color=color_list[k])
        h.set_label("CV fold {0}".format(k + 1))
        summaries_axes[0].set_xlabel("Iterations")
        summaries_axes[0].set_xlim((0, max_iter))
        summaries_axes[0].set_ylabel("Loss")
        summaries_axes[0].set_title("Learning curves")

    # Display the error rate across folds
    summaries_axes[1].bar(
        np.arange(1, K + 1), errors, color=color_list
    )
    summaries_axes[1].set_xlabel("Fold")
    summaries_axes[1].set_xticks(np.arange(1, K + 1))
    summaries_axes[1].set_ylabel("Error rate")
    summaries_axes[1].set_title("Test misclassification rates")

    # Display a diagram of the best network in last fold
    print("Diagram of best neural net in last fold:")
    weights = [net[i].weight.data.numpy().T for i in [0, 2]]
    biases = [net[i].bias.data.numpy() for i in [0, 2]]
    tf = [str(net[i]) for i in [1, 3]]
    draw_neural_net(weights, biases, tf)


if __name__ == "__main__":
    main()
