import numpy as np

def reduction(sample_F,n_modes):

    u, s, vh = np.linalg.svd(sample_F, full_matrices=False)

    u = u[:,:n_modes]

    return u.conj().T.dot(sample_F)
