import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
file_name = r"C:\Users\adity\OneDrive\Desktop\Sem 1\Intro to ML & DM\MC project-29th Feb'\seeds\seeds.txt"
df = pd.read_csv(file_name, sep='\t', header=None)
data = np.array(df.values)
attributeNames= [
        'area',
        'perimeter',
        'kernel len',
        'kernel width',
        'asymmetry',
        'kernel groove len',
        'Kama', 'Rosa', 'Canadian'
    ]

# Using area, perimeter, and kernel length to predict compactness
X = data[:, [0, 1, 3, 4, 5 , 6]]  # Features
Y = data[:, 2]  # Target variable - Compactness
N, M = X.shape


K = 10
# Initialize cross-validation
kf = KFold(n_splits=K, shuffle=True, random_state=1)

# Define lambda values to search
lambdas = np.logspace(-3, -1.5, 1000)

# Dictionary to store mean MSE for each lambda
mean_mse_values = {}
# Dictionary to store coefficients for each lambda
coefficients = {}

for lambda_ in lambdas:
    mse_values = []
    coeffs_for_lambda = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        # Standardization
        mu = np.mean(X_train, axis=0)
        sigma = np.std(X_train, axis=0)
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma

        # Standardize the target variable Y
        mu_Y = np.mean(y_train)
        sigma_Y = np.std(y_train)

        # Standardize Y in both training and testing sets
        y_train = (y_train - mu_Y) / sigma_Y
        y_test = (y_test - mu_Y) / sigma_Y


        # Train Linear Regression model with current lambda
        model = LinearRegression().fit(X_train, y_train)

        # Adjust coefficients with L2 regularization manually
        w_ridge = np.linalg.inv(X_train.T @ X_train + lambda_ * np.eye(X_train.shape[1])) @ X_train.T @ y_train

        # Calculate predictions
        y_pred = X_test @ w_ridge

        # Append coefficients for the current fold
        coeffs_for_lambda.append(w_ridge)

        # Calculate MSE
        mse = mean_squared_error(y_test, y_pred)
        mse_values.append(mse)

    # Store the mean MSE for the current lambda
    mean_mse_values[lambda_] = np.mean(mse_values)

    # Store the coefficients for the current lambda
    coefficients[lambda_] = np.mean(coeffs_for_lambda, axis=0)

# Find the lambda with the minimum mean MSE
best_lambda = min(mean_mse_values, key=mean_mse_values.get)

print("Optimal Lambda:", best_lambda)
print("Mean MSE:", mean_mse_values[best_lambda])

# Plot mean squared error vs. Lambda
plt.figure(figsize=(10, 8))
plt.plot(np.log10(list(mean_mse_values.keys())), list(mean_mse_values.values()), linestyle='-')
plt.scatter(np.log10([best_lambda]), mean_mse_values[best_lambda], color='red', marker='x', s=100)
plt.annotate(f'Optimal Lambda (approx): {best_lambda:.2f}', xy=(np.log10(best_lambda), mean_mse_values[best_lambda]), xytext=(-80, 30),
             textcoords='offset points', fontsize=12)
plt.xlabel('log10(Lambda)')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs. log10(Lambda)')
plt.grid(True)
plt.legend()
plt.show()


# Plot coefficients vs. Lambda
plt.figure(figsize=(10, 6))
for i in range(M):
    plt.plot(lambdas, [coeff[i] for coeff in coefficients.values()], marker='o', linestyle='-', label= attributeNames[i])

plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Coefficient Value')
plt.title('Change in Coefficients vs. Lambda')
plt.grid(True)
plt.legend()

# Print the coefficients
print("Coefficients for the optimal lambda:", coefficients[best_lambda])

# Print the attribute names and their corresponding coefficients
for i in range(M):
    print(f"{attributeNames[i]}: {coefficients[best_lambda][i]}")
