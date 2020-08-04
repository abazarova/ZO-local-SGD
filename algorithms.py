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


def noname_fun_one_to_all(fun, x0, grad, grad_params, method, communication_rounds, n, k, frq, lr, delta):
    res = minimize(fun, x0, method='BFGS', tol=1e-6)
    x = np.arange(0, communication_rounds, 1)
    y = []
    time_arr = []
    a = np.array([x0 for i in range(k)])
    t0 = time.time()
    for i in range(communication_rounds * frq):
        e = generate_vector(method, n)
        for j in range(k):
            a[j] -= grad(e, delta, a[j], grad_params[j]) * lr
        if i % (frq) == 0:
            avg = 0
            for j in range(k):
                avg += a[j]
            avg /= k
            for j in range(k):
                a[j] = avg
            if i == 0:
                fun0 = fun(a[0])
            y.append((fun(a[0]) - res.fun)/(fun0 - res.fun))
            time_arr.append(time.time() - t0)
    return x, y, time_arr
    
    
def noname_fun_one_to_one(fun, x0, grad, grad_params, method, communication_rounds, n, k, frq, lr, delta):
    res = minimize(fun, x0, method='BFGS', tol=1e-6)
    x = np.arange(0, communication_rounds, 1)
    y = []
    time_arr = []
    a = np.array([x0 for i in range(k)])
    t0 = time.time()
    for i in range(communication_rounds * frq):
        for j in range(k):
            e = generate_vector(method, n)
            a[j] -= grad(e, delta, a[j], grad_params[j]) * lr
        if i % (frq) == 0:
            avg = 0
            for j in range(k):
                avg += a[j]
            avg /= k
            for j in range(k):
                a[j] = avg
            if i == 0:
                fun0 = fun(a[0])
            y.append((fun(a[0]) - res.fun)/(fun0 - res.fun))
            time_arr.append(time.time() - t0)
    return x, y, time_arr

def zo_local_sgd_one_to_all(fun, x0, grad, params, method, n_batches, communication_rounds, n, k, frq, lr, delta):
    res = minimize(fun, x0, method='BFGS', tol=1e-6)
    arr_X = []
    arr_y = []
    for j in range(k):
        arr_X.append(params[j][0])
        arr_y.append(params[j][1])
    arr_X_split = [np.array_split(arr_X[i], n_batches) for i in range(k)]
    arr_y_split = [np.array_split(arr_y[i], n_batches) for i in range(k)]
    x = np.arange(0, communication_rounds, 1)
    y = []
    time_arr = []
    a = np.array([x0 for i in range(k)])
    t0 = time.time()
    for i in range(communication_rounds * frq):
        e = generate_vector(method, n)
        for j in range(k):
            q = random.randint(0, n_batches - 1)
            a[j] -= grad(e, delta, a[j], [arr_X_split[j][q], arr_y_split[j][q]]) * lr
        if i % (frq) == 0:
            avg = 0
            for j in range(k):
                avg += a[j]
            avg /= k
            for j in range(k):
                a[j] = avg
            if i == 0:
                fun0 = fun(a[0])
            y.append((fun(a[0]) - res.fun)/(fun0 - res.fun))
            time_arr.append(time.time() - t0)
    return x, y, time_arr

def zo_local_sgd_one_to_one(fun, x0, grad, params, method, n_batches, communication_rounds, n, k, frq, lr, delta):
    res = minimize(fun, x0, method='BFGS', tol=1e-6)
    arr_X = []
    arr_y = []
    for j in range(k):
        arr_X.append(params[j][0])
        arr_y.append(params[j][1])
    arr_X_split = [np.array_split(arr_X[i], n_batches) for i in range(k)]
    arr_y_split = [np.array_split(arr_y[i], n_batches) for i in range(k)]
    x = np.arange(0, communication_rounds, 1)
    y = []
    time_arr = []
    a = np.array([x0 for i in range(k)])
    t0 = time.time()
    for i in range(communication_rounds * frq):
        for j in range(k):
            e = generate_vector(method, n)
            q = random.randint(0, n_batches - 1)
            a[j] -= grad(e, delta, a[j], [arr_X_split[j][q], arr_y_split[j][q]]) * lr
        if i % (frq) == 0:
            avg = 0
            for j in range(k):
                avg += a[j]
            avg /= k
            for j in range(k):
                a[j] = avg
            if i == 0:
                fun0 = fun(a[0])
            y.append((fun(a[0]) - res.fun)/(fun0 - res.fun))
            time_arr.append(time.time() - t0)
    return x, y, time_arr

def gradient_descent(fun, w0, grad_params, communication_rounds, n, k, h, L):
    delta = 0.001
    lr = 1/L
    res = minimize(fun, w0, method='BFGS', tol=1e-6)
    x = np.arange(0, communication_rounds, 1)
    y = []
    time_arr = []
    w = np.array([w0 for i in range(k)])
    t0 = time.time()
    for i in range(communication_rounds * h):
        for j in range(k):
            w[j] -= logreg_grad(w[j], grad_params[j]) * lr
        if i % h == 0:
            avg = 0
            for j in range(k):
                avg += w[j]
            avg /= k
            for j in range(k):
                w[j] = avg
            if i == 0:
                fun0 = fun(w[0])
            y.append((fun(avg) - res.fun)/(fun0 - res.fun))
            time_arr.append(time.time() - t0)
    return x, y, time_arr

def s_gradient_descent(fun, w0, grad_params, n_batches, communication_rounds, n, k, h, L):
    delta = 0.001
    lr = 1/L
    res = minimize(fun, w0, method='BFGS', tol=1e-6)
    arr_X = []
    arr_y = []
    for j in range(k):
        arr_X.append(grad_params[j][0])
        arr_y.append(grad_params[j][1])
    arr_X_split = [np.array_split(arr_X[i], n_batches) for i in range(k)]
    arr_y_split = [np.array_split(arr_y[i], n_batches) for i in range(k)]
    x = np.arange(0, communication_rounds, 1)
    y = []
    time_arr = []
    w = np.array([w0 for i in range(k)])
    t0 = time.time()
    for i in range(communication_rounds * h):
        for j in range(k):
            q = random.randint(0, n_batches - 1)
            w[j] -= logreg_grad(w[j], [arr_X_split[j][q], arr_y_split[j][q]]) * lr
        if i % h == 0:
            avg = 0
            for j in range(k):
                avg += w[j]
            avg /= k
            for j in range(k):
                w[j] = avg
            if i == 0:
                fun0 = fun(w[0])
            y.append((fun(avg) - res.fun)/(fun0 - res.fun))
            time_arr.append(time.time() - t0)
    return x, y, time_arr
