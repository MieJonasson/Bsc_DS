import pandas as pd
import numpy as np
from collections import Counter

def k_neighbors_id(xs,new_x,k):
    dists = np.linalg.norm(xs - new_x, axis=1)
    return dists.argsort()[:k]

def predict(xs,ys,priors,k,new_x):
    ids = k_neighbors_id(xs,new_x,k)
    pred_ys = Counter(ys[ids])
    w_pred_ys = {y:pred_ys[y] * priors[y] for y in set(ys)}
    pred = list(set(ys))[0]
    for k in list(set(ys))[1:]:
        if w_pred_ys[k] > w_pred_ys[pred]:
            pred = k
    return pred

train = pd.read_csv('Exercises_7/Ex1-training.csv')
# test = pd.read_csv('Exercises_7/Ex1-test.csv')

priors = {1:0.0001, 2:0.02, 3:0.979}

new_data = pd.DataFrame([[5,5]],columns=['x1','x2'])

print(predict(train[['x1','x2']],train['y'],priors,5,new_data))