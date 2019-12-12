# Import the libraries
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score
# Import the dataset
dataset = pd.read_csv('Data.csv') #This dataset is just for me, just ignore it
X = dataset.iloc[:,1:8].values #Array of Independent Variables
Y = dataset.iloc[:, 8].values   #Array of Dependent Variables
Y = Y.reshape(500, 1)
mean_GRE = np.mean(X[:, 0])  #147.042500 #This is obtained from dataset.describe()
std_GRE = np.std(X[:, 0]) #85.854236 #This is obtained from dataset.describe()
mean_TOEFL = np.mean(X[:, 1]) #23.264000 #This is obtained from dataset.describe()
std_TOEFL = np.std(X[:, 1]) #14.846809 #This is obtained from dataset.describe()
mean_Ranking = np.mean(X[:, 2]) #30.554000 #This is obtained from dataset.describe()
std_Ranking = np.std(X[:, 2]) #21.778621 #This is obtained from dataset.describe()
mean_SOP = np.mean(X[:, 3]) #This is obtained from dataset.describe()
std_SOP = np.std(X[:, 3]) #This is obtained from dataset.describe()
mean_LOR = np.mean(X[:, 4]) #This is obtained from dataset.describe()
std_LOR = np.std(X[:, 4]) #This is obtained from dataset.describe()
mean_CGPA = np.mean(X[:, 5]) #This is obtained from dataset.describe()
std_CGPA = np.std(X[:, 5]) #This is obtained from dataset.describe()
mean_Research = np.mean(X[:, 6]) #This is obtained from dataset.describe()
std_Research = np.std(X[:, 6]) #This is obtained from dataset.describe()
stat = dataset.describe() #Reshaping Y as 200:1 matrix
X[:,0] = (X[:,0] - mean_GRE) / std_GRE #Normalizing TV column
X[:,1] = (X[:,1] - mean_TOEFL) / std_TOEFL  #Normalizing Radio column
X[:,2] = (X[:,2] - mean_Ranking) / std_Ranking #Normalizing Newspaper column
X[:,3] = (X[:,4] - mean_SOP) / std_SOP #Normalizing Newspaper column
X[:,4] = (X[:,4] - mean_LOR) / std_LOR #Normalizing Newspaper column
X[:,5] = (X[:,5] - mean_CGPA) / std_CGPA #Normalizing Newspaper column
X[:,6] = (X[:,6] - mean_Research) / std_Research #Normalizing Newspaper column
ones = np.ones((500, 1))
X = np.hstack((ones, X))
# Splitting the Dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)

def mserror(Y_pred, Y):
    a = (Y_pred-Y)
    a = np.square(a)
    b = a.sum()/500
    return b

def normal_equation(X, Y):
    P_inv = np.linalg.pinv(X, rcond=1e-16)
    m1 = np.matrix(P_inv)
    m2 = np.matrix(Y)
    weights = m1 * m2
    return weights

def linear_prediction(X, weights):
    a = np.matrix(X)
    b = np.matrix(weights)
    c = a * b
    return c

def stochastic_gradient_step(X, Y, w, train_ind, n = 0.01):
    Y_pred = linear_prediction(X, w)
    w0 = w[0,0] - ((2*n)/500) * (Y_pred[train_ind] - Y[train_ind])
    w1 = w[1,0] - ((2*n)/500) * X[train_ind, 1] * (Y_pred[train_ind] - Y[train_ind])
    w2 = w[2,0] - ((2*n)/500) * X[train_ind, 2] * (Y_pred[train_ind] - Y[train_ind])
    w3 = w[3,0] - ((2*n)/500) * X[train_ind, 3] * (Y_pred[train_ind] - Y[train_ind])
    w4=  w[4,0] - ((2*n)/500) * X[train_ind, 4] * (Y_pred[train_ind] - Y[train_ind])
    w5 = w[5,0] - ((2*n)/500) * X[train_ind, 5] * (Y_pred[train_ind] - Y[train_ind])
    w6 = w[6,0] - ((2*n)/500) * X[train_ind, 6] * (Y_pred[train_ind] - Y[train_ind])
    w7 = w[7,0] - ((2*n)/500) * X[train_ind, 7] * (Y_pred[train_ind] - Y[train_ind])
    w_new = np.array([w0, w1, w2, w3, w4, w5, w6, w7]).reshape(8, 1)
    return w_new

def stochastic_gradient_descent(X, Y, w_init, max_iter = 100000, eta = 0.01, seed = 42, verbose = "False"):
    a = np.array([]) 
    random.seed(seed)
    for i in range(max_iter):
         train_ind = np.random.randint(0, 399)
         W = stochastic_gradient_step(X, Y, w_init, train_ind, n = eta)
         w_init=W
         a = np.append(a, (mserror(linear_prediction(X, W), Y)))
    return W, a
  
         
w_zeros=np.array([[0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [0],
          [0]])
w_zeros.reshape(8, 1)
bek , qaba = stochastic_gradient_descent(X = X_train,Y = Y_train,w_init = w_zeros)
plt.figure(figsize=(6, 6))
plt.ylabel('MSE')
plt.xlabel('Iteration')
plt.plot(qaba)
print(bek)
print(qaba[99999])
    
Y_final = linear_prediction(X_test, bek)
print(math.sqrt(mserror(Y_final, Y_test)))
print(r2_score(y_true = Y_test, y_pred = Y_final))


