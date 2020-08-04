import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import random
import time
import pylab
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix

def quadratic_function(A, b, x):
    return 0.5 * x.T @ A @ x - b.T @ x 

def nesterov_function(x, a, L):
    return ((np.sum(np.diff((x - a)) ** 2) + (x[0] - a)** 2 + (x[x.shape[0] - 1]- a) ** 2) / 2 - (x[0] - a)) * L / 4

def logreg_function(w, X, y):
    l = np.log(1 + np.exp(-X.dot(w) * y))
    m = y.shape[0]
    return np.sum(l) / m

def pseudo_grad_quadratic(e, delta, x, params):
    A = params[0]
    b = params[1]
    theta = (quadratic_function(A, b, x + delta * e) - quadratic_function(A, b, x)) / delta
    return theta * e

def pseudo_grad_nesterov(e, delta, x, params):
    a = params[0]
    L = params[1]
    theta = (nesterov_function(x + delta * e, a, L) - nesterov_function(x, a, L)) / delta
    return theta * e

def pseudo_grad_logreg(e, delta, w, params):
    X = params[0]
    y = params[1]
    theta = (logreg_function(w + delta * e, X, y) - logreg_function(w, X, y)) / delta
    return theta * e
  
def logreg_sgrad(w, x_i, y_i):
    loss_sgrad = -y_i * x_i / (1 + np.exp(y_i * np.dot(x_i, w)))
    return loss_sgrad

def logreg_grad(w, X, y):
    loss_grad = np.mean([logreg_sgrad(w, X[i], y[i]) for i in range(len(y))], axis=0)
    return loss_grad
    
def generate_quadratic_functions(n, k, kappa):
    arr_A = []
    arr_b = []
    for i in range(k):
        kappa = 1000
        des = np.random.uniform(low = 1, high = kappa, size = n) 
        des = 1 + (kappa - 1) * (des - min(des)) / (max(des) - min(des))
        s = np.diag(des)
        q, _ = la.qr(np.random.rand(n, n))
        A = q.T @ s @ q
        b = np.random.rand(n)
        arr_A.append(A)
        arr_b.append(b)
    return arr_A, arr_b

def generate_logreg_dataset(file_name):
    X, y = load_svmlight_file(file_name)
    X = csr_matrix(X).toarray()
    if (file_name == 'mushrooms.txt'):
        y = -2 * y + 3
    return X, y

def generate_vector(generation_mode, n):
    if (generation_mode == 'random direction'):
        e = np.random.normal(size = n)
        e = e / la.norm(e)
    if (generation_mode == 'random coordinate'):
        e = np.zeros(n)
        q = random.randint(0, n - 1)
        e[q] = 1
    return e