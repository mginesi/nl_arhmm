import numpy as np

class Initial(object):

    def __init__(self, n_modes=3, density=None):
        self.n_modes = n_modes
        if density is None:
            density = np.random.rand(self.n_modes)
        if not(np.shape(density)[0] == self.n_modes):
            raise ValueError('initial.py: shape of initial density not compatible with number of modes!')

        # Making the vector an actual probability density
        density = np.maximum(density, 0.0)
        density /= np.sum(density)
        self.density = density

    def sample(self):
        # Method to sample an initial state from the density function
        cpdf = np.cumsum(self.density)
        rv = np.random.rand()
        return np.where(rv < cpdf)[-1][0]

if __name__ == '__main__':
    from initial import Initial
    import matplotlib.pyplot as plt
    in_d = Initial(3)
    sample_set = []
    for _i in range(500):
        sample_set.append(in_d.sample())
    plt.hist(sample_set)
    plt.title('Sample from ' + str(in_d.density))
    plt.show()
