import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
import torch
from dtuimldmtools import train_neural_net, visualize_decision_boundary, draw_neural_net
from scipy.io import loadmat
from sklearn import model_selection

from main import importData2


def main():
    data = importData2().standardized()
    X = data.X
    # y = np.array([[x] for x in np.array(data.classLabels, dtype=int).T])
    y = data.oneOutOfKEncodedClassLabels

    N, M = X.shape
    C = len(data.classNames)

    attributeNames = data.attributeNames
    classNames = data.classNames

    K = 2
    CV = model_selection.KFold(K, shuffle=True)

    # Setup figure for display of the decision boundary for the several crossvalidation folds.
    decision_boundaries = plt.figure(1, figsize=(10, 10))
    # Determine a size of a plot grid that fits visualizations for the chosen number
    # of cross-validation splits, if K=4, this is simply a 2-by-2 grid.
    subplot_size_1 = int(np.floor(np.sqrt(K)))
    subplot_size_2 = int(np.ceil(K / subplot_size_1))
    # Set overall title for all of the subplots
    plt.suptitle("Data and model decision boundaries", fontsize=20)
    # Change spacing of subplots
    plt.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=0.5, hspace=0.25)

    # Setup figure for display of learning curves and error rates in fold
    summaries, summaries_axes = plt.subplots(1, 2, figsize=(10, 5))
    # Make a list for storing assigned color of learning curve for up to K=10
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

    # Define the model structure
    n_hidden_units = 1  # number of hidden units in the single hidden layer
    # The lambda-syntax defines an anonymous function, which is used here to
    # make it easy to make new networks within each cross validation fold
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to H hiden units
        torch.nn.ReLU(),  # 1st transfer function
        # Output layer:
        # H hidden units to C classes
        # the nodes and their activation before the transfer
        # function is often referred to as logits/logit output
        torch.nn.Linear(n_hidden_units, C),  # C logits
        # To obtain normalised "probabilities" of each class
        # we use the softmax-funtion along the "class" dimension
        # (i.e. not the dimension describing observations)
        torch.nn.Softmax(dim=1),  # final tranfer function, normalisation of logit output
    )
    # Since we're training a neural network for binary classification, we use a
    # binary cross entropy loss (see the help(train_neural_net) for more on
    # the loss_fn input to the function)
    loss_fn = torch.nn.MSELoss()
    # Train for a maximum of 10000 steps, or until convergence (see help for the
    # function train_neural_net() for more on the tolerance/convergence))
    max_iter = 10000
    print("Training model of type:\n{}\n".format(str(model())))

    # Do cross-validation:
    errors = []  # make a list for storing generalizaition error in each loop
    # Loop over each cross-validation split. The CV.split-method returns the
    # indices to be used for training and testing in each split, and calling
    # the enumerate-method with this simply returns this indices along with
    # a counter k:
    for k, (train_index, test_index) in enumerate(CV.split(X, y)):
        print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))

        # Extract training and test set for current CV fold,
        # and convert them to PyTorch tensors
        X_train = torch.Tensor(X[train_index, :])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index, :])
        y_test = torch.Tensor(y[test_index])
        # y_test = y[test_index]

        net, final_loss, learning_curve = train_neural_net(
            model, loss_fn, X=X_train, y=y_train, n_replicates=3, max_iter=max_iter
        )

        print("\n\tBest loss: {}\n".format(final_loss))

        # Determine estimated class labels for test set
        y_softmax_logits = net(X_test)  # activation of final note, i.e. prediction of network
        y_test_est = (torch.max(y_softmax_logits, dim=1)[1]).data.numpy()
        y_test = y_test.type(dtype=torch.uint8)
        # Determine errors and error rate
        e = y_test_est != y_test
        error_rate = (sum(e).type(torch.float) / len(y_test)).data.numpy()
        # errors.append(error_rate)  # store error rate for current CV fold
        errors.append(np.mean(error_rate))  # store error rate for current CV fold

        # Make a subplot for current cross validation fold that displays the
        # decision boundary over the original data, "background color" corresponds
        # to the output of the sigmoidal transfer function (i.e. before threshold),
        # white areas are areas of uncertainty, and a deaper red/blue means
        # that the network "is more sure" of a given class.
        plt.figure(decision_boundaries.number)
        plt.subplot(subplot_size_1, subplot_size_2, k + 1)
        plt.title("CV fold {0}".format(k + 1), color=color_list[k])
        predict = lambda x: (
            torch.max(net(torch.tensor(x, dtype=torch.float)), dim=1)[1]
        ).data.numpy()

        plt.figure(1, figsize=(9, 9))
        visualize_decision_boundary(
            predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames
        )

        # Display the learning curve for the best net in the current fold
        (n_hidden_units_plot,) = summaries_axes[0].plot(learning_curve, color=color_list[k])
        n_hidden_units_plot.set_label("CV fold {0}".format(k + 1))
        summaries_axes[0].set_xlabel("Iterations")
        summaries_axes[0].set_xlim((0, max_iter))
        summaries_axes[0].set_ylabel("Loss")
        summaries_axes[0].set_title("Learning curves")

    # Display the error rate across folds
    summaries_axes[1].bar(
        np.arange(1, K + 1), np.squeeze(np.asarray(errors)), color=color_list
    )
    summaries_axes[1].set_xlabel("Fold")
    summaries_axes[1].set_xticks(np.arange(1, K + 1))
    summaries_axes[1].set_ylabel("Error rate")
    summaries_axes[1].set_title("Test misclassification rates")

    # Show the plots
    # plt.show(decision_boundaries.number) # try these lines if the following code fails (depends on package versions)
    # plt.show(summaries.number)
    plt.show()

    # Display a diagram of the best network in last fold
    print("Diagram of best neural net in last fold:")
    weights = [net[i].weight.data.numpy().T for i in [0, 2]]
    biases = [net[i].bias.data.numpy() for i in [0, 2]]
    tf = [str(net[i]) for i in [1, 3]]
    draw_neural_net(weights, biases, tf)

    # Print the average classification error rate
    print(
        "\nGeneralization error/average error rate: {0}%".format(
            round(100 * np.mean(errors), 4)
        )
    )

    print("Ran exercise 8.2.2.")


if __name__ == "__main__":
    main()
