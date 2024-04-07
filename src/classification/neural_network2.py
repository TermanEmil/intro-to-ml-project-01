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
    y = np.array(data.classLabels, dtype=int).T
    # y = data.oneOutOfKEncodedClassLabels

    # # Delete a class label
    # X = np.delete(X, y == 2, 0)
    # y = np.delete(y, y == 2)

    # Leave out only 2 attributes
    X = np.delete(X, [False, False, True, True, True, True, True], 1)

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
        X_train = X[train_index, :]
        y_train = y[train_index].squeeze()
        X_test = X[test_index, :]
        y_test = y[test_index].squeeze()

        n_hidden_units = 5  # number of hidden units in the signle hidden layer
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
        # Since we're training a multiclass problem, we cannot use binary cross entropy,
        # but instead use the general cross entropy loss:
        loss_fn = torch.nn.CrossEntropyLoss()
        # Train the network:
        net, _, _ = train_neural_net(
            model,
            loss_fn,
            X=torch.tensor(X_train, dtype=torch.float),
            y=torch.tensor(y_train, dtype=torch.long),
            n_replicates=3,
            max_iter=max_iter,
        )
        # Determine probability of each class using trained network
        softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
        # Get the estimated class as the class with highest probability (argmax on softmax_logits)
        y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy()
        # Determine errors
        e = y_test_est != y_test
        print(
            "Number of miss-classifications for ANN:\n\t {0} out of {1}".format(sum(e), len(e))
        )

        predict = lambda x: (
            torch.max(net(torch.tensor(x, dtype=torch.float)), dim=1)[1]
        ).data.numpy()
        plt.figure(1, figsize=(9, 9))
        visualize_decision_boundary(
            predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames
        )
        plt.title("ANN decision boundaries")

        plt.show()


if __name__ == "__main__":
    main()
