import numpy as np

class Dynamic(object):

    def __init__(self, n_dim, centers, widths, weights=None):
        '''
        Class encoding the dynamic.
        '''
        self.n_dim = n_dim
        self.centers = centers
        self.n_basis = np.shape(centers)[0]
        self.widths = widths
        if weights is None:
            weights = np.zeros([self.n_dim, self.n_basis])
        self.weights = weights
