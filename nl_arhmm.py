import numpy as np
import copy

from initial import Initial
from transition import Transition
from dynamic import Dynamic
from utils import normal_prob

class NL_ARHMM(object):

    def __init__(self, n_dim, n_modes, dyn_centers, dyn_widths, dyn_weights, sigmas):
        '''
        Class to implement Non-Linear Auto-Regressive Hidden Markov Models
        '''
        self.n_dim = n_dim
        self.n_modes = n_modes
        self.initial = Initial(self.n_modes)
        self.transition = Transition(self.n_modes)
        self.dynamics = []
        # TODO: default values for centers, widths and weights?
        for _ in range(self.n_modes):
            self.dynamics.append(Dynamic(self.n_dim, dyn_centers, dyn_widths, dyn_weights))
        self.sigma_set = sigmas

    def give_prob_of_next_step(self, y0, y1, mode):
        mu = self.dynamics[mode].apply_vector_field(y0)
        return normal_prob(y1, mu, self.sigma_set[mode])

    def simulate(self, y0=None, T=100):
        if y0 is None:
            y0 = np.zeros(self.n_dim)
        _mode = self.initial.sample()
        mode_seq = [_mode]
        _y = copy.deepcopy(y0)
        state_seq = [_y]
        for _t in range(T):
            _y = self.dynamics[_mode].apply_vector_field(_y)
            state_seq.append(_y)
            _mode = self.transition.sample(_mode)
            mode_seq.append(_mode)
        return [state_seq, mode_seq]
