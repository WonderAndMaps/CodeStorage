# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 13:01:51 2018

@author: fuyang
"""

import pandas as pd
import numpy as np
import time
from multiprocessing.dummy import Pool as ThreadPool

def square_euclidean_distances(X, Y):
    '''
    Considering the rows of X, Y as vectors, compute the
    distance matrix between each pair of vectors.
    Inspired by sklearn.
    '''
    XX = np.einsum('ij,ij->i', X, X)[:, np.newaxis]
    YY = np.einsum('ij,ij->i', Y, Y)[np.newaxis, :]
    distances = X.dot(Y.T)
    distances *= -2
    distances += XX
    distances += YY
    np.maximum(distances, 0, out=distances)
    return distances

def gaussian_rbf_kernel(X, Y, sigma=1.0):
    '''
    Compute the rbf (gaussian) kernel between X and Y:
        K(x, y) = exp(- ||x-y||^2/ sigma^2)
    for each pair of rows x in X and y in Y.
    '''
    K = square_euclidean_distances(X, Y)
    K /= -(sigma**2)
    np.exp(K, K)
    return K


def power_pca(A, k, n_iter=1000):
    '''
    Compute top n_eig eigenvalues and vectors of A simultaneously. 
    Return eigen value and vector in order.
        
    A: array of shape (n, n).
    k: int, number of top eigenvalues wanted.
    n_iter: int, number of iterations.
    ''' 
    n = A.shape[0]
    center = (np.eye(n) - np.ones([n,n])/n)
    A = center.dot(A).dot(center)
    Q = np.random.rand(n, k)
    Q, s = np.linalg.qr(Q)
    Qold = Q
 
    for i in range(n_iter):
        Z = A.dot(Q)
        Q, R = np.linalg.qr(Z)
        err = ((Q - Qold) ** 2).sum()
        Qold = Q
        
        if err < 1e-3:
            break

    return np.diag(R), Q


from sklearn.linear_model import LogisticRegression

def find_sigma(X, y, d, arg):
    
    K = gaussian_rbf_kernel(X, X, arg)
    lam, Z = power_pca(K, d)

    #fit softmax (no regulazation)
    clf = LogisticRegression(C=10**(20), multi_class='multinomial', solver='saga')
    clf.fit(Z, y)
    return 1-clf.score(Z, y)


#=======================body
#==============MLP
#L1
X = pd.read_csv('C:\\Users\\dell\\Desktop\\paper review\\mlpL1.txt',index_col=0).values
y = pd.read_csv('C:\\Users\\dell\\Desktop\\paper review\\mlpLabel.txt',index_col=0).values
y = np.apply_along_axis(np.argmax, 1, y)

#parallel
d = np.linspace(1,10,10).tolist()
sigma = np.linspace(10,50,20).tolist()
results1 = [0]*11
results1[0] = [0.9]*20

for i in d:
    i = int(i)
    time_start=time.time()
    pool = ThreadPool(2)
    results1[i] = pool.map(lambda x: find_sigma(X, y, i, arg=x), sigma)
    pool.close()
    pool.join()
    time_end=time.time()
    print('Time spent: ', time_end-time_start, 's')
    
pd.DataFrame([min(x) for x in results1]).to_csv('C:\\Users\\dell\\Desktop\\paper review\\mlpResultL1.txt')


#L2
X = pd.read_csv('C:\\Users\\dell\\Desktop\\paper review\\mlpL2.txt',index_col=0).values

sigma = np.linspace(20,100,20).tolist()
results2 = [0]*11
results2[0] = [0.9]*20

for i in d:
    i = int(i)
    time_start=time.time()
    pool = ThreadPool(2)
    results2[i] = pool.map(lambda x: find_sigma(X, y, i, arg=x), sigma)
    pool.close()
    pool.join()
    time_end=time.time()
    print('Time spent: ', time_end-time_start, 's')

pd.DataFrame([min(x) for x in results2]).to_csv('C:\\Users\\dell\\Desktop\\paper review\\mlpResultL2.txt')


#L0
X = pd.read_csv('C:\\Users\\dell\\Desktop\\paper review\\mlp0.txt',index_col=0).values

sigma = np.linspace(1,100,20).tolist()
results0 = [0]*11
results0[0] = [0.9]*20

for i in d:
    i = int(i)
    time_start=time.time()
    pool = ThreadPool(2)
    results0[i] = pool.map(lambda x: find_sigma(X, y, i, arg=x), sigma)
    pool.close()
    pool.join()
    time_end=time.time()
    print('Time spent: ', time_end-time_start, 's')

pd.DataFrame([min(x) for x in results0]).to_csv('C:\\Users\\dell\\Desktop\\paper review\\mlpResultL0.txt')

#=================CNN
#L1
X = pd.read_csv('C:\\Users\\dell\\Desktop\\paper review\\cnnL1.txt',index_col=0).values
y = pd.read_csv('C:\\Users\\dell\\Desktop\\paper review\\cnnLabel.txt',index_col=0).values
y = np.apply_along_axis(np.argmax, 1, y)

resultCnn1 = [0]*20
sigma = np.linspace(1,100,20).tolist()
i=0
        
for s in sigma:
    time_start=time.time()
    resultCnn1[i] = find_sigma(X, y, 10, arg=s)
    time_end=time.time()
    print('Time spent: ', time_end-time_start, 's')
    i += 1

    
pd.DataFrame([min(resultCnn1)]).to_csv('C:\\Users\\dell\\Desktop\\paper review\\cnnResultL1.txt')


#L2
X = pd.read_csv('C:\\Users\\dell\\Desktop\\paper review\\cnnL2.txt',index_col=0).values

resultCnn2 = [0]*20
sigma = np.linspace(1,100,20).tolist()
i=0
        
for s in sigma:
    time_start=time.time()
    resultCnn2[i] = find_sigma(X, y, 10, arg=s)
    time_end=time.time()
    print('Time spent: ', time_end-time_start, 's')
    i += 1

    
pd.DataFrame([min(resultCnn2)]).to_csv('C:\\Users\\dell\\Desktop\\paper review\\cnnResultL2.txt')


