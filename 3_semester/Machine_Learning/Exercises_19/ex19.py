# For learning a circular pattern we think having 
### 2 features -> hidden layer with 4 neurons -> output 
# is sufficient

# We have an input of 2 entries ---> we have a 4 x 2 weight matrix + 4 biases
## Now we see z = x . w + b (summed inputs) 
## and then a = activation_function(z)
### Now a becomes the new x for the next iterative step towards the output

### x will always be n x <number of features>
### w will always be <number of features> x <neurons in layer>
### b wil always be <number of neurons> x 1

# Loss function : L(x) = cross-entropy for binary probability : highest loss when 100% sure of opposite class

# Imports
import numpy as np
import matplotlib.pyplot as plt

# get data (syn3 from example dataset notebook)
def get_data(N):
    """ data(samples, features)"""
    data = np.empty(shape=(N,2), dtype = np.float32)  
    tar = np.empty(shape=(N,), dtype = np.float32) 
    N1 = int(2*N/3)
    
    # disk
    teta_d = np.random.uniform(0, 2*np.pi, N1)
    inner, outer = 3, 5
    r2 = np.sqrt(np.random.uniform(inner**2, outer**2, N1))
    data[:N1,0],data[:N1,1] = r2*np.cos(teta_d), r2*np.sin(teta_d)
        
    #circle
    teta_c = np.random.uniform(0, 2*np.pi, N-N1)
    inner, outer = 0, 2
    r2 = np.sqrt(np.random.uniform(inner**2, outer**2, N-N1))
    data[N1:,0],data[N1:,1] = r2*np.cos(teta_c), r2*np.sin(teta_c)
    

    tar[:N1] = np.ones(shape=(N1,))
    tar[N1:] = np.zeros(shape=(N-N1,))
    
    return data, tar

sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_der = lambda x: sigmoid(x) * (1 - sigmoid(x))
nll = lambda yhat, y: -(((y.T @ np.log(yhat))) + ((1-y).T @ np.log(1-yhat)))

def forward(x, wh, bh, wo, bo):
    # Pass to hidden layer
    zh = x @ wh + bh
    ah = sigmoid(zh)

    # Pass to output layer
    zo = ah @ wo + bo
    ao = sigmoid(zo)

    return zh, ah, zo, ao

def backward(x, y, wh, bh, wo, bo, lr = 10e-4):
    y = y.reshape(-1,1)
    # compute loss
    zh, ah, zo, ao = forward(x, wh, bh, wo, bo)

    # Output layer derivatives
    dl_dzo = ao - y # Loss derivative with respect to zo 
    dzo_dwo = ah

    dl_dwo = dzo_dwo.T @ dl_dzo # Relevant weight gradients
    dl_dbo = dl_dzo.sum(axis=0).reshape(1,-1) # Relevant bias gradient

    # Hidden layer derivatives
    dzo_dah = wo
    dl_dah = dl_dzo @ dzo_dah.T
    dah_dzh = sigmoid_der(zh)
    dl_dzh = dl_dah * dah_dzh
    dzh_dwh = x

    dl_dwh = dzh_dwh.T @ dl_dzh # Relevant weight gradients
    dl_dbh = dl_dzh.sum(axis=0).reshape(1,-1) # Relevant bias gradient

    # Updating weights and biases
    wh -= lr * dl_dwh
    bh -= lr * dl_dbh

    wo -= lr * dl_dwo
    bo -= lr * dl_dbo

    return wh, bh, wo, bo

def main():
    x, y = get_data(100)

    # Implementing Neural Network
    ## Initialise weights and biases nn - first hidden then output
    wh = np.random.random((2,4))
    bh = np.random.random((1,4))

    wo = np.random.random((4,1))
    bo = np.random.random((1,1))

    w = [wh, wo]
    b = [bh, bo]

    # Forward pass - prediction
    zh, ah, zo, ao = forward(x, wh, bh, wo, bo) # output ao (the prediction output) is n x 1

    # Training with Gradient Descent
    losses = [] 

    for epoch in range(1000):
        # compute gradients for loss with respect to all parameters
        # Update all weighte by rule w := w - lr * dl
        wh, bh, wo, bo = backward(x, y, wh, bh, wo, bo)
        loss = nll(forward(x, wh, bh, wo, bo)[3], y)[0]
        losses.append(loss)
        print(f' Epoch: {epoch + 1} \t Loss: {round(loss,4)} \t Accuracy:')
    
    plt.plot(range(1,1001), losses)
    plt.show()

if __name__ == '__main__':
    main()