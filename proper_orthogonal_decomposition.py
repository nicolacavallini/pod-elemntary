import numpy as np

import matplotlib.pyplot as plt

from local_gaussian_process import DataBase
from local_gaussian_process import GaussianProcess


def interpolate_coefficients(sample_t,C,C_,t,kernel_parameter,eigen_scale):

    C_new = np.zeros((0,t.shape[0]))

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

    return C_new
