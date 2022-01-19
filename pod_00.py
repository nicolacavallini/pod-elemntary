import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from matplotlib.gridspec import GridSpec

import GPy

from ezyrb import POD, RBF, Database, GPR
from ezyrb import ReducedOrderModel as ROM

if __name__ == "__main__":

    sech = lambda x : 1./np.cosh(x)

    y = np.linspace(-1,1,100)
    t_ids = np.arange(360,dtype=np.float64)

    [T_ids,Y] = np.meshgrid(t_ids,y)

    t = t_ids/359.
    T = T_ids/359.

    M = .2 * np.sin(2*np.pi*T)

    sigma = .05
    F = np.exp(-.5*((Y-M)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

    u, s, vh = np.linalg.svd(F, full_matrices=False)

    #plt
    #plt.show()

    modes = 20
    print("u.shape = "+str(u.shape))
    print(s.shape)
    print(vh.shape)

    u = u[:,:modes]

    C = u.conj().T.dot(F)

    print("C.shape = "+str(C.shape))

    reconstucted = u.dot(C)

    #fig, ax = plt.subplots(2,1)
    #
    #

    #plt.show()

    fig = plt.figure(constrained_layout=True,figsize=(12,4))

    #fig, ax = plt.subplots(2,1)
    gs = GridSpec(2, 8, figure=fig)

    ax0 = fig.add_subplot(gs[0,0:5])
    ax1 = fig.add_subplot(gs[1,0:5])
    ax2 = fig.add_subplot(gs[:,5:])
    ax2.yaxis.tick_right()

    ax0.pcolor(T,Y,F)
    ax1.pcolor(T,Y,reconstucted)

    ax2.plot(s/np.sum(s),'-o')

    plt.show()


    interval = 3
    t_sample_ids = np.arange(0,360,interval)

    t_sample_ids = np.reshape(t_sample_ids,(-1,1))

    #mask_T = np.isin(T_ids,t_sample_ids)
    mask_t = np.isin(t_ids,t_sample_ids)

    sample_t = np.reshape(t[mask_t],(-1,1))

    print("sample_t.shape ="+ str(sample_t.shape))
    print("F.shape ="+ str(F.shape))

    sample_F = F.T[mask_t]

    print("sample_F.shape ="+ str(sample_F.shape))
    #print(sample_T.shape)

    print(len(sample_t))
    print(len(sample_F))

    db = Database(sample_t, sample_F)
    ##pod = POD('svd')
    rbf = RBF()

    n_modes = 2

    pod = POD(method='svd',rank=n_modes)#RANK

    gpr = GPR()

    k = GPy.kern.RBF(input_dim=1, variance = .1, lengthscale=.1)
    rom = ROM(db, pod, gpr).fit(kern=k)
    #rom = ROM(db, pod, gpr).fit()
    #rom = ROM(db, pod, rbf).fit()

    new_F  = rom.predict(np.reshape(t,(-1,1)))

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
