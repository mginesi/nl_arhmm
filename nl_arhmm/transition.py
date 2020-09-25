import numpy as np
from nl_arhmm.utils import normalize_rows

class Transition(object):

    def __init__(self, n_modes=3, trans_mtrx=None):
        self.n_modes = n_modes
        if trans_mtrx is None:
            trans_mtrx = np.random.rand(self.n_modes, self.n_modes)
        if not(trans_mtrx.shape == (self.n_modes, self.n_modes)):
            raise ValueError('transition.py: shape of transition matrix not compatible with number of modes!')

        # Making each row of the matrix an actual probability density
        trans_mtrx = np.maximum(trans_mtrx, 0.0)
        trans_mtrx = normalize_rows(trans_mtrx)
        self.trans_mtrx = trans_mtrx
        self.logtrans = np.log(trans_mtrx)

    def sample(self, st):
        # Method to sample the next state given the actual state
        cpdf = np.cumsum(self.trans_mtrx[st])
        rv = np.random.rand()
        return np.where(rv < cpdf)[-1][0]

if __name__ == '__main__':
    from transition import Transition
    import matplotlib.pyplot as plt
    tr_d = Transition(3)
    st = 0
    sample_set = [st]
    for _i in range(50):
        st = tr_d.sample(st)
        sample_set.append(st)
    print(str(tr_d.trans_mtrx))
    plt.plot(sample_set, '.-')
    plt.title('Simulation the Markov chain')
    plt.show()