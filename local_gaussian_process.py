import numpy as np

import matplotlib.pyplot as plt

class DataBase:
    def __init__(self,x,fx):
        self.x = x.reshape(-1,1)
        self.fx = fx.reshape(-1,1)


def squared_exp(kp):

    def func(kernel_parameter,a,b):
        sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
        return np.exp(-.5 * (1./kernel_parameter) * sqdist)

    return lambda a,b : func(kp,a,b)

def radial_basis_function(kp):

    def func(kernel_parameter,a,b):
        A,B = np.meshgrid(a,b)
        return np.exp(-.5/kernel_parameter * (A-B)**2)

    return lambda a,b : func(kp,a,b)

class GaussianProcess:

    def kernel_selctor(self,kernel_name,kernel_parameter):
        available_kernels = {}
        available_kernels["squared-exp"] = squared_exp(kernel_parameter)

        return available_kernels[kernel_name]

    def __init__(self,kernel_name,kernel_parameter):

        print("kernel_parameter = "+str(kernel_parameter))

        self.kernel = self.kernel_selctor(kernel_name,kernel_parameter)

    def fit(self,db):

        self.db = db

        self.K = self.kernel(db.x, db.x)

        self.L = np.linalg.cholesky(self.K)

    def predict(self,x_test):

        K_star = self.kernel(self.db.x, x_test)
        print("K_star.shape = "+str(K_star.shape))
        print("K_star.shape = "+str(self.L.shape))

        Lk = np.linalg.solve(self.L, K_star)

        mu = np.dot(Lk.T, np.linalg.solve(self.L, self.db.fx))

        K_ = self.kernel(x_test, x_test)
        s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
        s = np.sqrt(s2)

        return mu,s
        #return 2,3
