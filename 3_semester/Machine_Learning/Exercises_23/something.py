import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

def main():
    iris = load_iris(as_frame=True)
    x, y = iris.data, iris.target
    x, y = x[['petal width (cm)', 'petal length (cm)']].to_numpy(), y.to_numpy()
    
    # Testing K-means for what k to choose
    max_k = 10
    deviances = []
    ks = range(1,max_k)

    for k in tqdm(ks):
        deviances.append(KMeans(n_clusters=k).fit(x).inertia_)
    
    #plt.plot(ks, deviances)
    #plt.show()

    # Fitting kmeans at chosen k=3
    clusters = KMeans(3).fit(x).labels_
    fig, ax = plt.subplots(figsize=(6,3), ncols=2)
    ax[0].scatter(x[:,0], x[:,1])
    ax[1].scatter(x[:,0], x[:,1])
    plt.show()

if __name__ == "__main__":
    main()