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
            weights = np.zeros([self.n_dim, self.n_basis + 1])
        self.weights = weights

    def compute_phi_vect(self, x):
        '''
        Compute the vector phi(x) which components are
          phi_0 = 1
          phi_j = exp (- alpha_j || x - mu_j || ^ 2)
        '''
        phi = np.ones(self.n_basis + 1)
        phi[1:] = np.exp(-self.widths * np.linalg.norm(self.centers - x, axis=1) ** 2.0)
        return phi

    def give_vector_field(self, x):
        return np.dot(self.weights, self.compute_phi_vect(x))