import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader # All pytorch functions want dataset objects of this type

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

class Data(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getindex__(self, i):
        return self.x[i], self.y[i]

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Specify the model architecture
        self.i2h = nn.Linear(2, 4) # Input to Hidden Layer - 2 features to 4 features
        self.h2o = nn.Linear(4, 1) # Hidden to Output Layer

        self.sigmoid = torch.sigmoid()
    
    def forward(self, x):
        zh = self.i2h(x)
        ah = self.sigmoid(zh)
        zo = self.h2o(ah)
        ao = self.sigmoid(zo)
        return ao

def main():
    # load data
    x, y = get_data(100)

    # Converting np.array to torch.tensor
    x, y = torch.tensor(x), torch.tensor(y)

    # Initialise Data class
    data = Data(x, y)
    loader = DataLoader(data, batch_size=2**5, shuffle=True)

    # Initialising the Neural Net
    neuralnet = NeuralNet()

if __name__ == '__main__':
    main()