import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import random
import time
import pylab
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from functions import *


def noname_fun_one_to_all(fun, x0, grad, grad_params, method, communication_rounds, n, k, h, lr, delta):
    res = minimize(fun, x0, method='BFGS', tol=1e-6)
    x = np.arange(0, communication_rounds, 1)
    y = []
    time_arr = []
    a = np.array([x0 for i in range(k)])
    fun0 = fun(x0)
    t0 = time.time()
    for i in range(communication_rounds * n * h):
        e = generate_vector(method, n)
        for j in range(k):
            a[j] -= grad(e, delta, a[j], grad_params[j]) * lr
        if i % (n * h) == 0:
            avg = 0
            for j in range(k):
                avg += a[j]
            avg /= k
            for j in range(k):
                a[j] = avg
            y.append((fun(a[0]) - res.fun)/(fun0 - res.fun))
            time_arr.append(time.time() - t0)
    return x, y, time_arr
    
    
def noname_fun_one_to_one(fun, x0, grad, grad_params, method, communication_rounds, n, k, h, lr, delta):
    res = minimize(fun, x0, method='BFGS', tol=1e-6)
    x = np.arange(0, communication_rounds, 1)
    y = []
    time_arr = []
    a = np.array([x0 for i in range(k)])
    fun0 = fun(x0)
    t0 = time.time()
    for i in range(communication_rounds * n * h):
        for j in range(k):
            e = generate_vector(method, n)
            a[j] -= grad(e, delta, a[j], grad_params[j]) * lr
        if i % (n * h) == 0:
            avg = 0
            for j in range(k):
                avg += a[j]
            avg /= k
            for j in range(k):
                a[j] = avg
            y.append((fun(a[0]) - res.fun)/(fun0 - res.fun))
            time_arr.append(time.time() - t0)
    return x, y, time_arr
'''

def logreg_func_one_to_all(generation_mode, communication_rounds, X, y, k, h):
    n = X.shape[1]
    L = 0.25 * np.max(np.sum(np.square(X), axis=1))
    arr_X = np.array_split(X, k)
    arr_y = np.array_split(y, k) 
    delta = 0.001
    lr = 1/L
    
    def sigma(w):
        sigm = 0
        for i in range(k):
            sigm += logreg_function(w, arr_X[i], arr_y[i])
        return sigm
    
    w0 = np.random.rand(n)
    res = minimize(sigma, w0, method='BFGS', tol=1e-6)
    x = np.arange(0, communication_rounds, 1)
    y = []
    time_arr = []
    w = np.zeros((k, n))
    sigma0 = sigma(np.zeros(n))
    t0 = time.time()
    for i in range(communication_rounds * n * h):
        e = generate_vector(generation_mode, n)
        for j in range(k):
            w[j] -= pseudo_grad_logreg(w[j], arr_X[j], arr_y[j], e, delta) * lr
        if i % (n * h) == 0:
            avg = 0
            for j in range(k):
                avg += w[j]
            avg /= k
            for j in range(k):
                w[j] = avg
            y.append((sigma(avg) - res.fun)/(sigma0 - res.fun))
            time_arr.append(time.time() - t0)
    return x, y, time_arr

def logreg_func_one_to_one(generation_mode, communication_rounds, X, y, k, h):
    n = X.shape[1]
    L = 0.25 * np.max(np.sum(np.square(X), axis=1))
    arr_X = np.array_split(X, k)
    arr_y = np.array_split(y, k) 
    delta = 0.001
    lr = 1/L
    
    def sigma(w):
        sigm = 0
        for i in range(k):
            sigm += logreg_function(w, arr_X[i], arr_y[i])
        return sigm
    
    w0 = np.random.rand(n)
    res = minimize(sigma, w0, method='BFGS', tol=1e-6)
    x = np.arange(0, communication_rounds, 1)
    y = []
    time_arr = []
    w = np.zeros((k, n))
    sigma0 = sigma(np.zeros(n))
    t0 = time.time()
    for i in range(communication_rounds * n * h):
        for j in range(k):
            e = generate_vector(generation_mode, n)
            w[j] -= pseudo_grad_logreg(w[j], arr_X[j], arr_y[j], e, delta) * lr
        if i % (n * h) == 0:
            avg = 0
            for j in range(k):
                avg += w[j]
            avg /= k
            for j in range(k):
                w[j] = avg
            y.append((sigma(avg) - res.fun)/(sigma0 - res.fun))
            time_arr.append(time.time() - t0)
    return x, y, time_arr

def gradient_descent(communication_rounds, X, y, k, h):
    n = X.shape[1]
    L = 0.25 * np.max(np.sum(np.square(X), axis=1))
    arr_X = np.array_split(X, k)
    arr_y = np.array_split(y, k) 
    delta = 0.001
    lr = 1/L
    
    def sigma(w):
        sigm = 0
        for i in range(k):
            sigm += logreg_function(w, arr_X[i], arr_y[i])
        return sigm
    
    x = np.arange(0, communication_rounds, 1)
    y = []
    time_arr = []
    w = np.zeros((k, n))
    sigma0 = sigma(np.zeros(n))
    t0 = time.time()
    for i in range(communication_rounds * h):
        for j in range(k):
            w[j] -= logreg_grad(w[j], arr_X[j], arr_y[j]) * lr
        if i % h == 0:
            avg = 0
            for j in range(k):
                avg += w[j]
            avg /= k
            for j in range(k):
                w[j] = avg
            y.append((sigma(avg) - res.fun)/(sigma0 - res.fun))
            time_arr.append(time.time() - t0)
    return x, y, time_arr
'''
