# LDA - all features gaussian with the same covariance matrix
# QDA - all features gaussian with individual covariance matrices
# Naive Bayes - eliminates assumption of gaussian features, BUT independent features still
### Implicit assumption: we model feature distributions for each class using Kernel Density Estimates

import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score

# import seaborn as sns
# import matplotlib.pyplot as plt

class NaiveBayesClassifier():
    """
    Interface to train and predict using Naive Bayes
    Models univariate class conditionals of continuous features using KDE
    Models univariate class conditionals of discrete features using Empirical PMF
    """
    def __init__(self):
        # Init class attributes
        self.priors = None # array like in dimension k (number of classes)
        self.class_conditionals = [] # list of list of callables (functions) in dimension k (classes) x p (features)
        self.idx2class = None
        self.class2idx = None

        # Convenience attributes
        self.x, self.y = None, None
        self.n, self.p, self.k = None, None, None

    def fit(self, x, y, feature_types):
        self.x = x
        self.y = y
        self.n, self.p = self.x.shape
        classes, counts = np.unique(self.y, return_counts=True)
        self.k = len(classes)
        self.idx2class = {i: classes[i] for i in range(self.k)}
        self.class2idx = {classes[i] : i for i in range(self.k)}

        # Fitting priors
        self.priors = counts / self.n

        # Fitting class conditionals
        for k in range(self.k):
            univariate_dists = []
            mask = self.y == self.idx2class[k]
            x_class_k = x[mask]

            for j in range(self.p):
                x_feature_j = x_class_k[:, j]
                feature_type = feature_types[j]

                if feature_type == 'discrete':
                    # Empirical PMF (add callable to dists)
                    univariate_dist = self._empirical_pmf(x_feature_j)
                    univariate_dists.append(univariate_dist)

                elif feature_type == 'continuous':
                    # model a kde (add callable to dists)
                    univariate_dist = self._kde(x_feature_j)
                    univariate_dists.append(univariate_dist)
                    pass

            self.class_conditionals.append(univariate_dists)

    def predict_proba(self, x):
        probs = [] # list of lists n x k
        for k in range(self.k):
            prior_k = self.priors[k]
            class_conditionals_k = np.ones(len(x))
            for j in range(self.p):
                class_conditional = self.class_conditionals[k][j](x[:,j])
                class_conditionals_k *= class_conditional

            probs.append(class_conditionals_k * prior_k)

        return np.array(probs).T


    def predict(self, x):
        return [self.idx2class[pred] for pred in np.argmax(self.predict_proba(x), axis=1)]

    def _kde(self, x):
        kde = KernelDensity().fit(x.reshape(-1,1))

        return lambda x: np.exp(kde.score_samples(x.reshape(-1,1)))

    def _empirical_pmf(self, x):
        # x is a vector of samples from a discrete feature
        pmf = {}
        values, counts = np.unique(x, return_counts=True)
        counts = counts / len(x)
        for val, count in zip(values, counts):
            pmf[val] = count
        
        return lambda x: np.array([pmf.get(sample, 0) for sample in x])

def main():
    # Loading data & restructure
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    xtrain, ytrain = train[['x1','x2','x3']].to_numpy(), train['y'].to_numpy()
    xtest, ytest = test[['x1','x2','x3']].to_numpy(), test['y'].to_numpy()

    # Doing pair plotting of train features
    # sns.pairplot(train, hue='y', corner=True)
    # plt.show()

    # Initialise clf
    clf = NaiveBayesClassifier()

    # Fit clf
    clf.fit(xtrain, ytrain, ['continuous', 'continuous', 'discrete'])

    # predict on test split
    # print(clf.predict_proba(xtest[:3]))
    # print(clf.predict(xtest[:3]))

    pred = clf.predict(xtest)
    print(f'Accuracy: {accuracy_score(pred, ytest)}')

if __name__ == '__main__':
    main()