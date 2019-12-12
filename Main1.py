#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 23:30:54 2019

@author: bekassyl
"""
EPSILON = 1e-10
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import math
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from scipy import stats
import statsmodels.api as sm
loocv = model_selection.LeaveOneOut()
plt.rcParams['figure.figsize'] = (8,6)
sns.set_style('white')
pd.set_option('expand_frame_repr',True)
pd.set_option('max_rows',80)
pd.set_option('expand_frame_repr',False)
data_path = '/Users/bekassyl/Desktop/BigData Project/'
data = pd.read_csv(data_path + 'Data.csv', encoding = 'utf-8')
print(data.head())
data.columns = [c.strip().lower().replace(" ","_") for c in data.columns]
data.info()
a = data.describe()
#sns.pairplot(data[['gre_score','toefl_score', 'cgpa', 'chance_of_admit']], kind='scatter', palette='Set1')
corrmat = data.drop('serial_no.', axis=1).corr()
print(corrmat)
X, y = shuffle(data.drop(['serial_no.','chance_of_admit'], axis=1), data.chance_of_admit, random_state= 5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
# fit scaler on training data
scaler = scaler.fit(X_train)
# transform training data using standard scaler
X_train_transformed = scaler.transform(X_train)
# transform test data fit scaler
X_test_transformed = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
# fit model to training data
linreg = linreg.fit(X_train_transformed, y_train)
#R^2 value of 
linreg_score_test = linreg.score(X_test_transformed, y_test)
linreg_score_train = linreg.score(X_train_transformed,y_train)
print("Linear Regression R^2 score on training set %.4f" %linreg_score_train)
print("Linear Regression R^2 score on test set     %.4f" %linreg_score_test)
pred = linreg.predict(X_test_transformed)
linreg_mse = mean_squared_error(y_pred=pred, y_true=y_test)
print("Linear Regression MSE on test set %.4f" %linreg_mse)
print("Linear Regression RMSE on test set %.4f" %math.sqrt(linreg_mse))
def rrse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Relative Squared Error """
    return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))
print("Linear Regression RRSE on training set %.4f" %rrse(y_train, linreg.predict(X_train_transformed)))
print("Linear Regression RRSE on test set %.4f" %rrse(y_test, pred))
def rae(actual: np.ndarray, predicted: np.ndarray):
    """ Relative Absolute Error (aka Approximation Error) """
    return np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual - np.mean(actual))) + EPSILON)
print("Linear Regression RAE on training set %.4f" %rae(y_train, linreg.predict(X_train_transformed)))
print("Linear Regression RAE on test set %.4f" %rae(y_test, pred))
linreg_coefs = linreg.coef_
coef_df = pd.DataFrame(data = list(zip(X_train.columns,linreg_coefs)), columns=['feature','coefficient'])
coef_df = coef_df.sort_values(by = 'coefficient', ascending=False)

print('***Linear Regression using 5-fold Cross Validationoss***')
scaler = preprocessing.StandardScaler()
linreg_pipe = Pipeline(steps=[('standardscaler', scaler ),('linear', LinearRegression())])
scores = cross_validate(return_train_score=True, error_score=True, estimator=linreg_pipe, X=X ,y=y, cv=5)
print("Average R^2:  %.4f" %scores['test_score'].mean())
print('***Linear Regression using Leave One Out Cross-Validation***')
scores = cross_validate(return_train_score=True, error_score=True, estimator=linreg_pipe, X=X ,y=y, cv=loocv)
print("Average R^2:  %.4f" %scores['train_score'].mean())
print('***Ridge Regression***')
ridge = Ridge()
alphas = np.logspace(-4, -.5, 30)
scores = list()
for alpha in alphas:
    ridge.alpha = alpha
    this_scores = cross_val_score(estimator = ridge, X=X, y=y, n_jobs=1, cv=10)
    scores.append(np.mean(this_scores))
plt.figure(figsize=(6, 6))
plt.semilogx(alphas, scores)
plt.ylabel('CV score')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='-', color='green')
plt.show()





