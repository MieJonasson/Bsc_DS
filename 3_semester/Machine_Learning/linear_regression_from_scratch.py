import numpy as np
from numpy import random
from numpy.linalg import inv
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine

# Helper
add_ones = lambda x: np.c_[x , np.ones(x.shape[0])]

#model function
linear_model = lambda x, beta: x @ beta
# x is the design matrix in nxp + 1 dimension (column of 1's on the right)
# beta: vector of weights

# Betas: the coefficients
get_best_beta = lambda x, y: inv(x.T @ x) @ (x.T @ y)

# Mean Squared Error Checking Function
mse = lambda y, y_hat: np.sum((y-y_hat)**2)

def main():
    # load data
    data = load_wine()
    columns = data.feature_names
    data = data.data

    alcohol_idx = 0
    color_idx = 9

    # Training data
    X = data[:, alcohol_idx].reshape(-1,1)
    Y = data[:, color_idx]

    # Training
    beta = get_best_beta(add_ones(X),Y)

    # Plotting Step
    _, ax = plt.subplots(figsize=(8,6))
    xs = np.linspace(X.min(), X.max(),100)
    predictions = linear_model(add_ones(xs), beta)
    ax.scatter(X,Y)
    ax.plot(xs, predictions, c='red')
    plt.show()

if __name__ == '__main__':
    main()