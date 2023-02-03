# Solutions Exercise 16
## Gradient Descent - We descend following the gradient direction until hitting a minimum!
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

f = lambda x: x**2 # Objective function
gradient = lambda x: 2*x # Gradient function for objective function

def old_main():
    x = np.linspace(-5,5,100)
    y = f(x)

    plt.plot(x,y)
    # plt.show()

    # Gradient Descent
    x = np.random.uniform(-5,5)
    alpha = 0.1 # learning rate - usually pretty small value
    for i in range(100):
        # Gradient step in opposite direction of gradient
        x -= alpha * gradient(x)
        print(f'iteration {i + 1} with x: {x}')

# Helper functions v2
add_ones = lambda x: np.c_[x, np.ones(x.shape[0])]
predict = lambda x, w: x @ w
mse = lambda x, w, y: np.mean((y-predict(add_ones(x), w))**2)
sse = lambda x, w, y: (predict(add_ones(x),w)-y).T @ (predict(add_ones(x),w)-y)
sse_gradient = lambda x, w, y: - (add_ones(x).T @ (y - predict(add_ones(x),w)))

def main():
    wine = load_wine()
    data = wine.data
    feature_names = wine.feature_names
    features = ['alcohol', 'color_intensity']

    # idx for x & y
    x_idx, y_idx = [i for i in range(len(feature_names)) if feature_names[i] in features]
    x = data[:, x_idx].reshape(-1,1)
    y = data[:,y_idx]
    n = len(x)

    # Scaling features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # plt.scatter(x,y)
    # plt.show()

    w = np.random.random(2)

    # print(predict(add_ones(x), w))
    # print(mse(x,w,y))

    # Batch Gradient Descent
    print('Batch Gradient Descent:')
    epochs = 50 # Epoch means i have looked at the entire dataset
    alpha = 0.001
    loss_history = []
    for epoch in range(epochs):
        # Compute gradient vector
        w -= alpha * sse_gradient(x,w,y)
        loss = sse(x,w,y)
        loss_history.append(loss)
        print(f'After Epoch {epoch + 1} with Sum of Squared Errors loss: {loss}')
    
    # Plotting the loss over each iteration
    fig, axes = plt.subplots(1, 2, figsize = (15,7))
    axes[0].plot(list(range(epochs)), loss_history)

    # Stochastic Gradient Descent
    print('\nStochastic Gradient Descent:')
    epochs = 50 # Epoch means i have looked at the entire dataset
    alpha = 0.001
    loss_history = []
    for epoch in range(epochs):
        for _ in range(n):
            # Choosing a random point to base our update on - like bootstrapping
            idx = int(np.random.random() * n)
            x_samp = x[idx].reshape(-1,1)
            y_samp = y[idx].reshape(-1)

            # Compute gradient vector
            w -= alpha * sse_gradient(x_samp,w,y_samp)

        loss = sse(x,w,y)
        loss_history.append(loss)
        print(f'After Epoch {epoch + 1} with Sum of Squared Errors loss: {loss}')
    
    # Plotting the loss over each iteration
    axes[1].plot(list(range(epochs)), loss_history)
    plt.show()

if __name__ == '__main__':
    main()