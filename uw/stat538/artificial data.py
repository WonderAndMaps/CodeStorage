# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 18:50:23 2018

@author: dell
"""
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import math
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



#===========1
X1, y1 = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=0)
X1[y1==0,0] -= max(X1[y1==0,0])+.5
X1[y1==1,0] += abs(min(X1[y1==1,0]))+.5
X1[y1==0,1] -= np.mean(X1[y1==0,1])
X1[y1==1,1] -= np.mean(X1[y1==1,1])
X1[:,1] *= np.log(1+abs(X1[:,0])) 



#===========2
def spiral(radius, step, resolution=.1, angle=0.0, start=0.0):
    dist = start+0.0
    coords=[]
    while dist*math.hypot(math.cos(angle),math.sin(angle))<radius:
        cord=[]
        cord.append(dist*math.cos(angle))
        cord.append(dist*math.sin(angle))
        coords.append(cord)
        dist+=step
        angle+=resolution
    return coords

sp = np.array(spiral(3,.03))
X21 = np.empty([0,2])
X22 = np.empty([0,2])

for i in range(5):
    X21 = np.concatenate((X21, (np.array([sp[:,0]*.75, sp[:,1]*4]).T + np.random.normal(0,0.05,size=sp.shape))[:50,:]))
    X22 = np.concatenate((X22, (np.array([-sp[:,0]*.75-0.15, -sp[:,1]*4-1.25]).T + np.random.normal(0,0.05,size=sp.shape))[:50,:]))

X2 = np.concatenate((X21,X22))
y2 = np.array([1]*X21.shape[0] + [0]*X22.shape[0])

#==============3
X31 = np.empty([0,2])
X32 = np.empty([0,2])

for i in range(5):
    X31 = np.concatenate((X31, (np.array([sp[:,0]*.5, sp[:,1]*3.5]).T + np.random.normal(0,0.05,size=sp.shape))[:90,:]))
    X32 = np.concatenate((X32, (np.array([-sp[:,0]*.5, -sp[:,1]*3.5-1.25]).T + np.random.normal(0,0.05,size=sp.shape))[:90,:]))

X3 = np.concatenate((X31,X32))
y3 = np.array([1]*X31.shape[0] + [0]*X32.shape[0])

#====================body

d = np.linspace(1,10,10).tolist()
sigma = np.linspace(1,100,20).tolist()
results1 = [0]*11
results1[0] = [0.5]*20

for i in d:
    i = int(i)
    time_start=time.time()
    pool = ThreadPool(2)
    results1[i] = pool.map(lambda x: find_sigma(X1, y1, i, arg=x), sigma)
    pool.close()
    pool.join()
    time_end=time.time()
    print('Time spent: ', time_end-time_start, 's')


sigma = np.linspace(1,10,20).tolist()
results2 = [0]*11
results2[0] = [0.5]*20

for i in d:
    i = int(i)
    time_start=time.time()
    pool = ThreadPool(2)
    results2[i] = pool.map(lambda x: find_sigma(X2, y2, i, arg=x), sigma)
    pool.close()
    pool.join()
    time_end=time.time()
    print('Time spent: ', time_end-time_start, 's')


sigma = np.linspace(3,10,20).tolist()
results3 = [0]*11
results3[0] = [0.5]*20

for i in d:
    i = int(i)
    time_start=time.time()
    pool = ThreadPool(2)
    results3[i] = pool.map(lambda x: find_sigma(X3, y3, i, arg=x), sigma)
    pool.close()
    pool.join()
    time_end=time.time()
    print('Time spent: ', time_end-time_start, 's')

plt.figure(figsize=(15,10))

plt.subplot(231)
plt.scatter(X1[y1==0,0],X1[y1==0,1],color='r',s=10)
plt.scatter(X1[y1==1,0],X1[y1==1,1],color='b',s=10)
plt.xticks([-5, 5],['',''])
plt.yticks([-10, 10],['',''])

plt.subplot(232)
plt.scatter(X21[:,0],X21[:,1],color='r',s=10)
plt.scatter(X22[:,0],X22[:,1],color='b',s=10)
plt.xticks([-1.5, 1.5],['',''])
plt.yticks([-7, 5.75],['',''])

plt.subplot(233)
plt.scatter(X31[:,0],X31[:,1],color='r',s=10)
plt.scatter(X32[:,0],X32[:,1],color='b',s=10)
plt.xticks([-1.5, 1.5],['',''])
plt.yticks([-7, 5.75],['',''])

plt.subplot(234)
plt.plot([i for i in range(11)], [min(x) for x in results1], linewidth=1, color=[.2,.2,.2])
plt.xlabel('Number of principal components')
plt.ylabel('Misclassification rate')

plt.subplot(235)
plt.plot([i for i in range(11)], [min(x) for x in results2], linewidth=1, color=[.2,.2,.2])
plt.xlabel('Number of principal components')
plt.ylabel('Misclassification rate')

plt.subplot(236)
plt.plot([i for i in range(11)], [min(x) for x in results3], linewidth=1, color=[.2,.2,.2])
plt.xlabel('Number of principal components')
plt.ylabel('Misclassification rate')
plt.show()



