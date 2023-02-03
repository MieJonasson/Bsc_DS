# exercise_05.py
# linear regression model of mpg tilde horsepower

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

from matplotlib import pyplot as plt

np.random.seed(1)

mse = lambda y, y_hat : np.mean((y-y_hat)**2)

def kfolds(formula, data, k=5):
    # Get entire data data
    total_mse = 0.0
    # Split into k folds (randomized fashion)
    indices = np.random.permutation(len(data))
    data = data.iloc[indices]
    splits = np.array_split(indices, k)
    # Train K different models
    for i in range(k):
        # Split into train-test for ith fold
        test_data = data.iloc[splits[i]]
        train_data = data.iloc[[x for x in range(len(data)) if x not in splits[i]]]

        # Fit model on train
        model = smf.ols(formula, train_data).fit()

        # Performance on Held out data (ith fold)
        preds = model.predict(test_data)
        split_mse = mse(preds, test_data['mpg'])
        
        total_mse += split_mse

    # k mse values
    return total_mse / k

def main():
    # load data
    data = pd.read_csv('auto.csv')

    # split data into train/test
    ### Test data is to be completely untouched at ALL TIMES
    indices = np.random.permutation(len(data))
    train_idx, test_idx = indices[:len(indices)//2], indices[len(indices)//2:]
    train, test = data.iloc[train_idx], data.iloc[test_idx]
    # sklearn.model_selection.train_test_split(X,y)

    # scatter
    #plt.scatter(train['horsepower'],train['mpg'],alpha=0.7, c='red',label='Training Samples')
    #plt.scatter(test['horsepower'],test['mpg'],alpha=0.7,c='blue',label='Test samples')
    #plt.legend(loc='best')
    #plt.show()

    # Fitting a Linear Regression model
    lm = smf.ols("mpg ~ 1 + horsepower", train).fit()
    # lm.summary()
    qm = smf.ols("mpg ~ 1 + horsepower + I(horsepower**2)", train).fit()
    cm = smf.ols("mpg ~ 1 + horsepower + I(horsepower**2) + I(horsepower**3)", train).fit()

    # Performance on train and test - Linear
    ltrain_preds = lm.fittedvalues
    ltest_preds = lm.predict(test)

    print('\nLinear')
    print(f"Training MSE: {mse(ltrain_preds, train['mpg'])}")
    print(f"Test MSE: {mse(ltest_preds, test['mpg'])}")

    # Performance on train and test - Quadratic
    qtrain_preds = qm.fittedvalues
    qtest_preds = qm.predict(test)

    print('\nQuadratic')
    print(f"Training MSE: {mse(qtrain_preds, train['mpg'])}")
    print(f"Test MSE: {mse(qtest_preds, test['mpg'])}")

    # Performance on train and test - Cubic
    ctrain_preds = cm.fittedvalues
    ctest_preds = cm.predict(test)

    print('\nCubic')
    print(f"Training MSE: {mse(ctrain_preds, train['mpg'])}")
    print(f"Test MSE: {mse(ctest_preds, test['mpg'])}")

    # What happens if we train on the entire dataset?
    lm = smf.ols("mpg ~ 1 + horsepower", data).fit()
    qm = smf.ols("mpg ~ 1 + horsepower + I(horsepower**2)", data).fit()
    cm = smf.ols("mpg ~ 1 + horsepower + I(horsepower**2) + I(horsepower**3)", data).fit()

    print('\nModel Fitted on all data points:')
    print(f'Linear Model AIC: {lm.aic}')
    print(f'Quadratic Model AIC: {qm.aic}')
    print(f'Cubic Model AIC: {cm.aic}')

    # k-fold Cross-Validation
    estimate_mse = kfolds("mpg ~ 1 + horsepower", data)
    print('\nK-fold Cross Validation')
    print(f"Estimated MSE: {estimate_mse}")

if __name__=='__main__':
    main()