import numpy as np
import copy

from nl_arhmm.initial import Initial
from nl_arhmm.transition import Transition
from nl_arhmm.dynamic import GRBF_Dynamic
from nl_arhmm.dynamic import Linear_Dynamic
from nl_arhmm.utils import normal_prob, normalize_vect, normalize_rows, normalize_mtrx

class ARHMM(object):

    def __init__(self, n_dim, n_modes, dynamics, sigmas):
        '''
        Class to implement Non-Linear Auto-Regressive Hidden Markov Models.
        '''
        self.n_dim = n_dim
        self.n_modes = n_modes
        self.initial = Initial(self.n_modes)
        self.transition = Transition(self.n_modes)
        self.dynamics = dynamics
        self.sigma_set = sigmas

    def compute_likelihood(self, data_stream): # FIXME
        '''
        Compute the likelihood
          p ( X | Theta ) = sum_{Z} p ( X | Z , Theta )
        by scratch.
        '''
        alpha_stream = self.compute_forward_var(data_stream)
        return self.give_likelihood(alpha_stream)

    def give_likelihood(self, alpha_stream):
        '''
        Compute the likelihood of the data
          p ( X | Theta ) = sum_{Z} p ( X | Z , Theta )
        '''
        return np.sum(alpha_stream[-1])

    def em_algorithm(self, data_set, tol = 0.05, max_iter = 10,verbose=True):
        '''
        Perform the EM algorithm.
        '''
        # Check if data_set is a single demonstration or a list
        if not(isinstance(data_set, list)):
            data_set = [data_set]

        # Perform EM algorithm
        count = 0
        new_lh = self.compute_likelihood(data_set)
        print('Step 0: LH = ' + str(new_lh))
        convergence = False
        while not convergence:
            count += 1
            old_lh = copy.deepcopy(new_lh)
            new_lh = self.em_step(data_set)
            convergence = (((new_lh - old_lh) / old_lh) < tol) or (count > max_iter)
            if verbose:
                print('Step ' + str(count) + ': LH = ' + str(new_lh))

    def em_step(self, data_stream): # FIXME
        '''
        Performs a step of the EM algorithm.
        '''
        # Compute forward and backward variables
        alpha_stream, c_rescale_stream = self.compute_forward_var(data_stream)
        beta_stream = self.compute_backward_var(data_stream, c_rescale_stream)

        # Compute gamma and xi functions
        gamma_stream = self.compute_gamma(alpha_stream, beta_stream)
        xi_stream = self.compute_xi(alpha_stream, beta_stream, data_stream, c_rescale_stream)

        # Maximize
        self.maximize_initial(gamma_stream)
        self.maximize_transition(gamma_stream, xi_stream)
        self.maximize_emissions(gamma_stream, data_stream)

        # This re-computation is needed to compute the new likelihood
        alpha_stream = self.compute_forward_var(data_stream)

        return self.give_likelihood(alpha_stream)

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

    def simulate(self, y0=None, T=100, noise=None):
        if y0 is None:
            y0 = np.zeros(self.n_dim)
        if noise is None:
            noise = np.eye(self.n_dim)
        _mode = self.initial.sample()
        mode_seq = [_mode]
        _y = copy.deepcopy(y0)
        state_seq = [_y]
        for _t in range(T):
            _y = self.dynamics[_mode].apply_vector_field(_y) + \
                np.dot(noise, np.random.randn(self.n_dim))
            state_seq.append(_y)
            _mode = self.transition.sample(_mode)
            mode_seq.append(_mode)
        _y = self.dynamics[_mode].apply_vector_field(_y)
        state_seq.append(_y)
        return [np.asarray(state_seq), mode_seq]

    def viterbi(self, data_stream):
        '''
        Applies the Viterbi algorithm to compute the most probable sequence of states given
        the observations.
        '''
        # Initialize the variables
        T = len(data_stream) - 1
        T_1 = np.zeros([T, self.n_modes])
        T_2 = np.zeros([T, self.n_modes])
        
        # Compute the basis of recursion
        for _s in range(self.n_modes):
            T_1[0][_s] = self.give_prob_of_next_step(data_stream[0], data_stream[1], _s) * \
                self.initial.density[_s]
            # T_2[0] = 0.0

        # Recursion over time
        # FIXME: is it possible to vectorize something?
        for _t in range(1, T):
            for _s in range(self.n_modes):
                to_maximize = T_1[_t - 1] * self.transition.trans_mtrx[:, _s] * \
                    self.give_prob_of_next_step(data_stream[_t], data_stream[_t + 1], _s)
                _max = np.max(to_maximize)
                _argmax = np.where(to_maximize == _max)[0][0]
                T_1[_t, _s] = copy.deepcopy(_max)
                T_2[_t, _s] = copy.deepcopy(_argmax)

        # Extract the most probable sequence
        z = np.zeros(T)
        max_T = np.max(T_1[T - 1])
        argmax_T = np.where(T_1[T - 1] == max_T)[0][0]
        z[-1] = int(argmax_T)
        for _t in range(T - 1, 1, -1):
            z[_t - 1] = T_2[_t, int(z[_t])]
        return z

    # --------------------------------------------------------------------------------------- #
    #                              Expectation step functions                                 #
    # --------------------------------------------------------------------------------------- #

    def compute_forward_var(self, data_stream):  # FIXME
        '''
        Recursively compute the forward variables
          alpha(z_t) = p (y_0, .. , y_{t+1}, z_t | Theta^{old})
        '''
        # Initialization
        T = len(data_stream) - 1
        alpha = np.zeros([T, self.n_modes])
        c_rescale = np.zeros(T)

        # Basis of recursion
        for _m in range(self.n_modes):
            alpha[0][_m] = normal_prob(data_stream[1],
                self.dynamics[_m].apply_vector_field(data_stream[0]), self.sigma_set[_m]) * \
                self.initial.density[_m]
        c_rescale[0] = np.sum(alpha[0])
        alpha[0] /= c_rescale[0]

        # Recursion
        p_future = np.zeros(self.n_modes) # initialization
        for _t in range(1, T):
            # Computing p(y_{t+1} | y_t, z_t)
            for _m in range(self.n_modes):
                p_future[_m] = normal_prob(data_stream[_t+1],
                    self.dynamics[_m].apply_vector_field(data_stream[_t]), self.sigma_set[_m])
            alpha[_t] = p_future * np.dot(np.transpose(self.transition.trans_mtrx),
                alpha[_t - 1])
            c_rescale[_t] = np.sum(alpha[_t])
            alpha[_t] /= c_rescale[_t]

        return [alpha, c_rescale]

    def compute_backward_var(self, data_stream, c_rescale_stream):  # FIXME
        '''
        Recursively compute the backward variables
          beta(z_t) = p (y_{t+2}, .. , y_T | y_0, .. , y_{t+1}, z_t, Theta^{old})
        '''
        # Initialization
        T = len(data_stream) - 1
        beta = np.zeros([T, self.n_modes])

        # Basis of recursion
        beta[T - 1] = np.ones(self.n_modes)

        # Recursion
        p_future = np.zeros(self.n_modes) # initialization
        for _t in range(T - 2, -1, -1):
            # Computing p(y_{t+2} | y_{t+1}, z_{t+1})
            for _m in range(self.n_modes):
                p_future[_m] = normal_prob(data_stream[_t + 2],
                    self.dynamics[_m].apply_vector_field(data_stream[_t + 1]),
                    self.sigma_set[_m])
            beta[_t] = np.dot(self.transition.trans_mtrx,
                beta[_t + 1] * p_future) / c_rescale_stream[_t + 1]

        return beta

    # --------------------------------------------------------------------------------------- #
    #                              Maximization step functions                                #
    # --------------------------------------------------------------------------------------- #

    def compute_gamma(self, alpha, beta): # FIXME
        return normalize_rows(alpha * beta)

    def compute_xi(self, alpha, beta, data, c_rescale): # FIXME
        T = np.shape(alpha)[0]
        xi = np.zeros([T - 1, self.n_modes, self.n_modes])
        p_future = np.zeros(self.n_modes) # FIXME: computed in other functions!
        for _t in range(T - 1):
            for _m in range(self.n_modes):
                p_future[_m] = normal_prob(data[_t + 2],
                    self.dynamics[_m].apply_vector_field(data[_t + 1]), self.sigma_set[_m])
            xi[_t] = self.transition.trans_mtrx * (beta[_t + 1] * p_future) * \
                np.reshape(alpha[_t], [self.n_modes, 1]) / c_rescale[_t + 1]
            xi[_t] = normalize_mtrx(xi[_t])
        return xi

    def maximize_initial(self, gamma): # FIXME
        self.initial.density = normalize_vect(gamma[0])

    def maximize_transition(self, gamma, xi): # FIXME
        num = np.sum(xi, axis=0)
        den = np.sum(gamma[:-1], axis=0)
        self.transition.trans_mtrx = normalize_rows(num / np.reshape(den, [self.n_modes, 1]))

    def maximize_emissions(self, gamma, data_stream): # FIXME
        in_data = data_stream[:-1]
        out_data = data_stream[1:]
        T = len(out_data)
        out_data = np.asarray(out_data)
        phi_in_data = []
        # The phi map does not depend on the hidden mode, so we can compute it before the loop.
        for _, _in in enumerate(in_data):
            phi_in_data.append(self.dynamics[0].compute_phi_vect(_in))
        dim_phi_data = phi_in_data[0].size
        for _m in range(self.n_modes):
            gamma_m = gamma[:, _m]

            # Update of the covariance matrix
            expected_out_data = []
            for _, _in in enumerate(in_data):
                expected_out_data.append(self.dynamics[_m].apply_vector_field(_in))
            expected_out_data = np.asarray(expected_out_data)
            err = np.reshape(out_data - expected_out_data, [T, self.n_dim, 1])
            err_t = np.reshape(out_data - expected_out_data, [T, 1, self.n_dim])
            cov_err = np.matmul(err, err_t)
            self.sigma_set[_m] = np.sum(cov_err * np.reshape(gamma_m, [T, 1, 1]), 0) / \
                np.sum(gamma_m)

            # Update of the matrix of weights
            out_data_3d = np.reshape(np.asarray(out_data), [T, self.n_dim, 1])
            phi_in_row = np.reshape(np.asarray(phi_in_data), [T, 1, dim_phi_data])
            phi_in_column = np.reshape(np.asarray(phi_in_data), [T, dim_phi_data, 1])
            num = np.sum(
                np.reshape(gamma_m, [T, 1, 1]) * np.matmul(out_data_3d, phi_in_row), 0)
            den = np.sum(
                np.reshape(gamma_m, [T, 1, 1]) * np.matmul(phi_in_column, phi_in_row), 0)
            self.dynamics[_m].weights = np.dot(num, np.linalg.pinv(den))

class GRBF_ARHMM(ARHMM):

    def __init__(self, n_dim, n_modes, dyn_center, dyn_widths, dyn_weights, sigmas):
        '''
        Class to implement Non-Linear Auto-Regressive Hidden Markov Models.
        '''
        self.n_dim = n_dim
        self.n_modes = n_modes
        self.initial = Initial(self.n_modes)
        self.transition = Transition(self.n_modes)
        self.dynamics = []
        for _m in range(self.n_modes):
            self.dynamics.append(GRBF_Dynamic(self.n_dim, dyn_center[_m], dyn_widths[_m], dyn_weights[_m]))
        self.sigma_set = sigmas
        self.model = ARHMM(self.n_dim, self.n_modes, self.dynamics, self.sigma_set)

class Linear_ARHMM(ARHMM):

    def __init__(self, n_dim, n_modes, dyn_mtrxs, sigmas):
        '''
        Class to implement Non-Linear Auto-Regressive Hidden Markov Models.
        '''
        self.n_dim = n_dim
        self.n_modes = n_modes
        self.initial = Initial(self.n_modes)
        self.transition = Transition(self.n_modes)
        self.dynamics = []
        for _m in range(self.n_modes):
            self.dynamics.append(Linear_Dynamic(self.n_dim, dyn_mtrxs[_m]))
        self.sigma_set = sigmas
        self.model = ARHMM(self.n_dim, self.n_modes, self.dynamics, self.sigma_set)