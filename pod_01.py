import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from matplotlib.gridspec import GridSpec

import GPy

from ezyrb import POD, RBF, Database, GPR
from ezyrb import ReducedOrderModel as ROM

import proper_orthogonal_decomposition as pod

def plot_reconstruction():

    fig = plt.figure(constrained_layout=True,figsize=(12,4))

    gs = GridSpec(2, 8, figure=fig)

    ax0 = fig.add_subplot(gs[0,0:5])
    ax1 = fig.add_subplot(gs[1,0:5])
    ax2 = fig.add_subplot(gs[:,5:])
    ax2.yaxis.tick_right()

    ax0.pcolor(T,Y,F)
    ax1.pcolor(T,Y,reconstucted)

    spectrum = s/np.sum(s)

    ax2.bar(np.arange(n_modes),spectrum[:n_modes],color='r')
    ax2.bar(np.arange(n_modes,spectrum.shape[0]),spectrum[n_modes:],color='b')

    plt.show()

def plot_pod_result():

    fig = plt.figure(constrained_layout=True,figsize=(12,4))

    #fig, ax = plt.subplots(2,1)
    gs = GridSpec(2, 8, figure=fig)

    ax0 = fig.add_subplot(gs[0,0:5])
    ax1 = fig.add_subplot(gs[1,0:5])
    ax2 = fig.add_subplot(gs[:,5:])
    ax2.yaxis.tick_right()

    ax0.set_title("POD result")
    ax0.pcolor(T,Y,F)
    ax1.pcolor(T,Y,new_F.T)

    error = np.abs((new_F.T-F).flatten())
    ax2.hist(error)

    plt.show()




if __name__ == "__main__":

    y = np.linspace(-1,1,100)
    t_ids = np.arange(360,dtype=np.float64)

    [T_ids,Y] = np.meshgrid(t_ids,y)

    t = t_ids/359
    T = T_ids/359

    M = .4 * np.sin(2*np.pi*T)

    sigma = .1
    F = np.exp(-.5*((Y-M)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

    u, s, vh = np.linalg.svd(F, full_matrices=False)

    n_modes = 10

    u = u[:,:n_modes]

    C_ = u.conj().T.dot(F)

    reconstucted = u.dot(C_)

    plot_reconstruction()

    interval = 10
    t_sample_ids = np.arange(0,360,interval)

    t_sample_ids = np.reshape(t_sample_ids,(-1,1))

    #mask_T = np.isin(T_ids,t_sample_ids)
    mask_t = np.isin(t_ids,t_sample_ids)

    sample_t = np.reshape(t[mask_t],(-1,1))

    sample_F = F.T[mask_t].T

    u, eigenvals, vh = np.linalg.svd(sample_F, full_matrices=False)

    u = u[:,:n_modes]

    C = u.conj().T.dot(sample_F)

    eigen_scale = eigenvals[:n_modes]/eigenvals[0]

    plt.plot(eigen_scale)
    plt.show()

    kernel_parameter = .005

    C_new = pod.interpolate_coefficients(sample_t,C,C_,t,kernel_parameter,eigen_scale)


    F_new = u.dot(C_new)

    #print(sample_T.shape)

    print(len(sample_t))
    print(len(sample_F))

    #db = Database(sample_t, sample_F)
    ###pod = POD('svd')
    #rbf = RBF()

    #pod = POD(method='svd',rank=n_modes)#RANK

    #gpr = GPR()

    ##k = GPy.kern.RBF(input_dim=1, variance = .1, lengthscale=.1)
    ##rom = ROM(db, pod, gpr).fit(kern=k)
    #rom = ROM(db, pod, gpr).fit()
    ##rom = ROM(db, pod, rbf).fit()

    #new_F  = rom.predict(np.reshape(t,(-1,1)))

    #plot_pod_result()
