import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from main import importData2

random_state = 1


def main():
    data = importData2().standardized()
    X = data.X
    y = data.classLabels

    classNames = data.classNames

    things_to_delete = [
        [False, False, False, False, False, False, False],
        [True, True, True, True, True, True, False],
        [True, True, True, True, True, False, True],
        [True, True, True, True, False, True, True],
        [True, True, True, False, True, True, True],
        [True, True, False, True, True, True, True],
        [True, False, True, True, True, True, True],
        [False, True, True, True, True, True, True],
        [False, True, True, True, False, True, False],
    ]

    for thing_to_delete in things_to_delete:
        # Delete some attributes to see how the training behaves
        X = np.delete(np.array(data.X, copy=True), thing_to_delete, 1)
        attributeNames = np.delete(np.array(data.attributeNames, copy=True), thing_to_delete, 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)

        opt_lambda = 12.505385872903888

        model = LogisticRegression(
            random_state=random_state,
            multi_class="multinomial", solver='lbfgs',
            max_iter=20000,
            penalty="l2",
            C=1 / opt_lambda
        )
        y_score = model.fit(X_train, y_train).decision_function(X_test)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot of a ROC curve for a specific class
        plt.figure()
        for i in range(3):
            plt.plot(fpr[i], tpr[i], label=f'{classNames[i]} (area = {roc_auc[i]:0.4})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot([0, 1], [0, 1], 'k--')

        attributeNamesJoin = ', '.join(attributeNames)
        plt.title(f'ROC curve using {attributeNamesJoin}')
        plt.legend(loc="lower right")
        plt.show()


if __name__ == '__main__':
    main()
