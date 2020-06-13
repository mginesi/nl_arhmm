import numpy as np
import copy

from initial import Initial
from transition import Transition
from dynamic import Dynamic
from utils import normal_prob, normalize_vect, normalize_rows, normalize_mtrx

class NL_ARHMM(object):

    def __init__(self, n_dim, n_modes, dyn_centers, dyn_widths, dyn_weights, sigmas):
        '''
        Class to implement Non-Linear Auto-Regressive Hidden Markov Models.
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

    def compute_likelihood(self, alpha_stream, beta_stream):
        '''
        Compute the likelihood of the data
          p ( X | Theta ) = sum_{Z} p ( X | Z , Theta )
        '''
        return np.sum(alpha_stream[-1])

    def em_step(self, data_stream):
        '''
        Performs a step of the EM algorithm.
        '''
        # Compute forward and backward variables
        alpha_stream = self.compute_forward_var(data_stream)
        beta_stream = self.compute_backward_var(data_stream)

        # Compute gamma and xi functions
        gamma_stream = self.compute_gamma(alpha_stream, beta_stream)
        xi_stream = self.compute_xi(alpha_stream, beta_stream, data_stream)

        # Maximize
        self.maximize_initial(gamma_stream)
        self.maximize_transition(gamma_stream, xi_stream)
        self.maximize_emissions(gamma_stream, data_stream)

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
        beta[T - 1] = np.ones(self.n_modes)

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

    # --------------------------------------------------------------------------------------- #
    #                              Maximization step functions                                #
    # --------------------------------------------------------------------------------------- #

    def compute_gamma(self, alpha, beta):
        return normalize_rows(alpha * beta)

    def compute_xi(self, alpha, beta, data):
        T = np.shape(alpha)[0] + 1
        xi = np.zeros([T - 1, self.n_modes, self.n_modes])
        p_future = np.zeros(self.n_modes) # FIXME: computed in other functions!
        for _t in range(T - 2):
            for _m in range(self.n_modes):
                p_future[_m] = normal_prob(data[_t + 2],
                    self.dynamics[_m].apply_vector_field(data[_t + 1]), self.sigma_set[_m])
            xi[_t] = self.transition.trans_mtrx * (beta[_t] * p_future) * \
                np.reshape(alpha[_t], [self.n_modes, 1])
        return xi

    def maximize_initial(self, gamma):
        self.initial.density = normalize_vect(gamma[0])

    def maximize_transition(self, gamma, xi):
        num = np.sum(xi, axis=0)
        den = np.sum(gamma[:-1], axis=0)
        self.transition.trans_mtrx = normalize_rows(num / np.reshape(den, [self.n_modes, 1]))

    def maximize_emissions(self, gamma, data_stream):
        in_data = data_stream[:-1]
        out_data = data_stream[1:]
        for _m in self.n_modes:
            self.dynamics[_m].learn_vector_field(in_data, out_data, gamma[:, _m])