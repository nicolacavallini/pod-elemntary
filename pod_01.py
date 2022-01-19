import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from matplotlib.gridspec import GridSpec

import GPy

from ezyrb import POD, RBF, Database, GPR
from ezyrb import ReducedOrderModel as ROM

from local_gaussian_process import DataBase
from local_gaussian_process import GaussianProcess

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

    spectrum = eigenvals/np.sum(eigenvals)

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

    u, eigenvals, vh = np.linalg.svd(F, full_matrices=False)


    n_modes = 10
    print("u.shape = "+str(u.shape))
    print(eigenvals.shape)
    print(vh.shape)

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

    print("sample_t.shape ="+ str(sample_t.shape))
    print("F.shape ="+ str(F.shape))

    sample_F = F.T[mask_t].T

    u, s, vh = np.linalg.svd(sample_F, full_matrices=False)

    u = u[:,:n_modes]

    #C_new = np.zeros()
    C = u.conj().T.dot(sample_F)

    C_new = np.zeros((0,t.shape[0]))

    eigen_scale = eigenvals[:n_modes]/eigenvals[0]

    plt.plot(eigen_scale)
    plt.show()

    kernel_parameter = .005


    for id, (c, l) in enumerate(zip(C,eigen_scale)):

        db = DataBase(sample_t,c)

        gp = GaussianProcess("squared-exp",kernel_parameter*l)
        gp.fit(db)
        mu, s = gp.predict(t.reshape(-1,1))

        C_new = np.vstack((C_new,mu.flatten()))

        fig, ax = plt.subplots(1,2)

        ax[0].plot(sample_t,c,'x')
        ax[0].plot(t,mu)
        ax[0].plot(t,C_[id])

        ax[1].imshow(gp.kernel(sample_t,sample_t))
        plt.show()

    print("C_new.shape = "+str(C_new.shape))
    print("sample_F.shape = "+str(sample_F.shape))
    print("u.shape = "+str(u.shape))


    F_new = u.dot(C_new)

    print("F_new.shape = "+str(F_new.shape))



    plt.imshow(F_new)
    plt.show()

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
