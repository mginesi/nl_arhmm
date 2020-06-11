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

    def e_step(self, data_stream):
        '''
        Perform the Expectation step
        '''
        alpha_stream = self.compute_forward_var(data_stream)
        beta_stream = self.compute_backward_var(data_stream)
        return [alpha_stream, beta_stream]

    def give_prob_of_next_step(self, y0, y1, mode):
        mu = self.dynamics[mode].apply_vector_field(y0)
        return normal_prob(y1, mu, self.sigma_set[mode])

    def set_up_dynamic_guess(self, in_data, out_data):
        '''
        Set up a guess for the weights by dividing the input and output set in m equal
        segments (m being the number of modes).
        '''
        t_seg = int(len(in_data) / self.n_modes)
        for _m in range(self.n_modes):
            _in_set = in_data[_m * t_seg:(_m + 1) * t_seg]
            _out_set = out_data[_m * t_seg:(_m + 1) * t_seg]
            self.dynamics[_m].learn_vector_field(_in_set, _out_set)
            self.sigma_set[_m] = self.dynamics[_m].estimate_cov_mtrx(_in_set, _out_set)

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

    # --------------------------------------------------------------------------------------- #
    #                              Expectation step functions                                 #
    # --------------------------------------------------------------------------------------- #

    def compute_forward_var(self, data_stream):
        '''
        Recursively compute the forward variables
          alpha(z_t) = p (y_0, .. , y_{t+1}, z_t | Theta^{old})
        '''
        # Initialization
        T = len(data_stream) - 1
        alpha = np.zeros([self.n_modes, T])

        # Basis of recursion
        for _m in range(self.n_modes):
            alpha[0][_m] = normal_prob(data_stream[1],
                self.dynamics[_m].apply_vector_field(data_stream[0]), self.sigma_set[_m]) * \
                self.initial.density

        # Recursion
        p_future = np.zeros(self.n_modes) # initialization
        for _t in range(1, T):
            # Computing p(y_{t+1} | y_t, z_t)
            for _m in range(self.n_modes):
                p_future[_m] = normal_prob(data_stream[_t+1],
                    self.dynamics[_m].apply_vector_field(data_stream[_t]), self.sigma_set[_m])
            alpha[_t] = p_future * np.dot(np.transpose(self.transition.trans_mtrx),
                alpha[_t - 1])

        return alpha

    def compute_backward_var(self, data_stream):
        '''
        Recursively compute the backward variables
          beta(z_t) = p (y_{t+2}, .. , y_T | y_0, .. , y_{t+1}, z_t, Theta^{old})
        '''
        # Initialization
        T = len(data_stream) - 1
        beta = np.zeros([self.n_modes, T])

        # Basis of recursion
        for _m in range(self.n_modes):
            beta[T - 1][_m] = normal_prob(data_stream[T],
                self.dynamics[_m].apply_vector_field(data_stream[T - 1]), self.sigma_set[_m])

        # Recursion
        p_future = np.zeros(self.n_modes) # initialization
        for _t in range(T - 2, 0, -1):
            # Computing p(y_{t+2} | y_{t+1}, z_{t+1})
            for _m in range(self.n_modes):
                p_future[_m] = normal_prob(data_stream[_t + 2],
                    self.dynamics[_m].apply_vector_field(data_stream[_t + 1]),
                    self.sigma_set[_m])
            beta[_t] = np.dot(self.transition.trans_mtrx,
                beta[_t + 1] * p_future)

        return beta