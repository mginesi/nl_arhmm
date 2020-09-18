import numpy as np
from scipy.special import logsumexp
import copy

import multiprocessing

from nl_arhmm.initial import Initial
from nl_arhmm.transition import Transition
from nl_arhmm.dynamic import GRBF_Dynamic
from nl_arhmm.dynamic import Linear_Dynamic
from nl_arhmm.utils import normal_prob, log_normal_prob, normalize_vect, normalize_rows, normalize_mtrx

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

    def compute_log_likelihood(self, data_stream):
        '''
        Compute the likelihood
          p ( X | Theta ) = sum_{Z} p ( X | Z , Theta )
        by scratch.
        '''
        pool = multiprocessing.Pool()
        forward_stream = pool.map(self.compute_forward_var, data_stream)
        c_stream = [forward_stream[i][1] for i in range(len(forward_stream))]
        return self.give_log_likelihood(c_stream)

    def give_log_likelihood(self, log_c_stream):
        '''
        Compute the likelihood of the data
        '''
        lkl = 0
        for _log_c in log_c_stream:
            lkl += np.sum(_log_c)
        return lkl

    def em_algorithm(self, data_set, tol = 0.05, max_iter = 10,verbose=True):
        '''
        Perform the EM algorithm.
        '''
        # Check if data_set is a single demonstration or a list
        if not(isinstance(data_set, list)):
            data_set = [data_set]

        # Perform EM algorithm
        count = 0
        new_lh = self.compute_log_likelihood(data_set)
        print('Step 0: LH = ' + str(new_lh))
        convergence = False
        while not convergence:
            count += 1
            old_lh = copy.deepcopy(new_lh)
            new_lh = self.em_step(data_set)
            convergence = ((np.abs((new_lh - old_lh) / old_lh) < tol)) or (count > max_iter)
            if verbose:
                print('Step ' + str(count) + ': LH = ' + str(new_lh))

    def em_step(self, data_stream):
        '''
        Performs a step of the EM algorithm.
        '''
        pool = multiprocessing.Pool()
        # Compute forward and backward variables
        forward_stream = pool.map(self.compute_forward_var, data_stream)
        alpha_stream = [forward_stream[i][0] for i in range(len(forward_stream))]
        c_rescale_stream = [forward_stream[i][1] for i in range(len(forward_stream))]
        beta_stream = pool.map(self.compute_backward_var, zip(data_stream, c_rescale_stream))

        # Compute gamma and xi functions
        gamma_stream = pool.map(self.compute_gamma, zip(alpha_stream, beta_stream))
        xi_stream = pool.map(self.compute_xi, zip(alpha_stream, beta_stream, data_stream, c_rescale_stream))

        # Maximize
        self.maximize_initial(gamma_stream)
        self.maximize_transition(gamma_stream, xi_stream)
        self.maximize_emissions(gamma_stream, data_stream)
        
        self.initial.loginit = np.log(self.initial.density)
        self.transition.logtrans = np.log(self.transition.trans_mtrx)

        # This re-computation is needed to compute the new likelihood
        forward_stream = pool.map(self.compute_forward_var, data_stream)
        alpha_stream = [forward_stream[i][0] for i in range(len(forward_stream))]

        return self.give_log_likelihood(alpha_stream)

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

    def compute_forward_var(self, _data):
        '''
        Recursively compute the (logarithm of) scaled forward variables and the scaling factors
          alpha(z_t) = p (z_t | y_0, .. , y_{t+1}, Theta^{old})
          c_t = p (y_t | y_0, ..., y_{t-1}, Theta^{\old})
        '''
        T = len(_data) - 1
        log_alpha = np.zeros([T, self.n_modes])
        log_c_rescale = np.zeros(T)
        # Basis of recursion
        for _m in range(self.n_modes):
            log_alpha[0][_m] = log_normal_prob(_data[1],
                self.dynamics[_m].apply_vector_field(_data[0]), self.sigma_set[_m]) + \
                self.initial.density[_m]
        log_c_rescale[0] = logsumexp(log_alpha[0])
        log_alpha[0] -= log_c_rescale[0]

        # Recursion
        log_p_future = np.zeros(self.n_modes) # initialization
        for _t in range(1, T):
            # Computing p(y_{t+1} | y_t, z_t)
            for _m in range(self.n_modes):
                log_p_future[_m] = log_normal_prob(_data[_t+1],
                    self.dynamics[_m].apply_vector_field(_data[_t]), self.sigma_set[_m])
            log_alpha[_t] = log_p_future + \
                logsumexp(log_alpha[_t - 1] + np.transpose(self.transition.logtrans), axis = 1)
            log_c_rescale[_t] = logsumexp(log_alpha[_t])
            log_alpha[_t] -= log_c_rescale[_t]

        return [log_alpha, log_c_rescale]

    def compute_backward_var(self, _in):
        '''
        Recursively compute the (logarithm of the) scaled backward variables
                          p (y_{t+1}, .. , y_T | y_t, z_t, Theta^{old})
          beta(z_t) = ----------------------------------------------------
                       p (y_{t+1}, .. , y_T | y_0, ..., y_t, Theta^{old})
        '''
        _data = _in[0]
        _log_c_rescale = _in[1]
        # Initialization
        T = len(_data) - 1
        log_beta = np.zeros([T, self.n_modes])

        # Recursion
        log_p_future = np.zeros(self.n_modes) # initialization
        for _t in range(T - 2, -1, -1):
            # Computing p(y_{t+2} | y_{t+1}, z_{t+1})
            for _m in range(self.n_modes):
                log_p_future[_m] = log_normal_prob(_data[_t + 2],
                    self.dynamics[_m].apply_vector_field(_data[_t + 1]),
                    self.sigma_set[_m])
            log_beta[_t] = logsumexp(log_beta[_t + 1] + log_p_future + self.transition.logtrans,
                                     axis = 1) - _log_c_rescale[_t + 1]

        return log_beta

    def compute_gamma(self, _in):
        _log_alpha = _in[0]
        _log_beta = _in[1]
        return normalize_rows(np.exp(_log_alpha + _log_beta))

    def compute_xi(self, _in):
        log_alpha = _in[0]
        log_beta = _in[1]
        data = _in[2]
        log_c_rescale = _in[3]
        T = np.shape(log_alpha)[0]
        _xi = np.zeros([T - 1, self.n_modes, self.n_modes])
        log_p_future = np.zeros(self.n_modes) # FIXME: computed in other functions!
        for _t in range(T - 1):
            for _m in range(self.n_modes):
                log_p_future[_m] = log_normal_prob(data[_t + 2],
                    self.dynamics[_m].apply_vector_field(data[_t + 1]), self.sigma_set[_m])
            _xi[_t] = (log_alpha[_t] + self.transition.logtrans.transpose()).transpose() + \
                log_p_future + log_beta[_t + 1] - log_c_rescale[_t + 1]
            _xi[_t] = normalize_mtrx(np.exp(_xi[_t]))
        return _xi

    # --------------------------------------------------------------------------------------- #
    #                              Maximization step functions                                #
    # --------------------------------------------------------------------------------------- #


    def maximize_initial(self, gamma_set):
        new_init = np.zeros(self.n_modes)
        K = len(gamma_set)
        for _, _gamma in enumerate(gamma_set):
            new_init += normalize_vect(_gamma[0]) / K
        self.initial.density = normalize_vect(new_init)

    def maximize_transition(self, gamma_set, xi_set):
        num = np.zeros([self.n_modes, self.n_modes])
        den = np.zeros(self.n_modes)
        for k in range(len(gamma_set)):
            num += np.sum(xi_set[k], axis=0)
            den += np.sum(gamma_set[k][:-1], axis=0)
        self.transition.trans_mtrx = normalize_rows(num / np.reshape(den, [self.n_modes, 1]))

    def maximize_emissions_components(self,_in):
        gamma = _in[0]
        data = _in[1]
        in_data = data[:-1]
        out_data = data[1:]
        T = len(out_data)
        out_data = np.asarray(out_data)
        cov_num = [0 for _s in range(self.n_modes)]
        cov_den = [0 for _s in range(self.n_modes)]
        weights_num = [0 for _s in range(self.n_modes)]
        weights_den = [0 for _s in range(self.n_modes)]
        for _s in range(self.n_modes):
            phi_in_data = []
            # FIXME can be parallelized
            for _, _in in enumerate(in_data):
                phi_in_data.append(self.dynamics[_s].compute_phi_vect(_in))
            dim_phi_data = phi_in_data[0].size
            # Covariance Matrix "update"
            expected_out_data = []
            for _, _in in enumerate(in_data):
                expected_out_data.append(self.dynamics[_s].apply_vector_field(_in))
            expected_out_data = np.asarray(expected_out_data)
            _gamma = gamma[:, _s]
            err = np.reshape(out_data - expected_out_data, [T, self.n_dim, 1])
            err_t = np.reshape(out_data - expected_out_data, [T, 1, self.n_dim])
            cov_err = np.matmul(err, err_t)
            cov_num[_s] = np.sum(cov_err * np.reshape(_gamma, [T, 1, 1]), 0)
            cov_den[_s] = np.sum(_gamma)

            # Dynamics' weight "update"
            out_data_3d = np.reshape(np.asarray(out_data), [T, self.n_dim, 1])
            phi_in_row = np.reshape(np.asarray(phi_in_data), [T, 1, dim_phi_data])
            phi_in_column = np.reshape(np.asarray(phi_in_data), [T, dim_phi_data, 1])
            weights_num[_s] = np.sum(
                np.reshape(_gamma, [T, 1, 1]) * np.matmul(out_data_3d, phi_in_row), 0)
            weights_den[_s] = np.sum(
                np.reshape(_gamma, [T, 1, 1]) * np.matmul(phi_in_column, phi_in_row), 0)
        return [weights_num, weights_den, cov_num, cov_den]

    def maximize_emissions(self, gamma_set, data_set):
        # Initialize the needed arrays (which sum over k)
        # For each mode we need two matrices for the weights,
        # and a matrix and a scalar for the covariance
        pool = multiprocessing.Pool()
        _out = pool.map(self.maximize_emissions_components, zip(gamma_set, data_set))
        weights_num = [_out[i][0] for i in range(len(_out))]
        weights_den = [_out[i][1] for i in range(len(_out))]
        cov_num = [_out[i][2] for i in range(len(_out))]
        cov_den = [_out[i][3] for i in range(len(_out))]
        omega_num = [np.zeros([self.n_dim, self.dynamics[0].n_basis + 1])
                        for s in range(self.n_modes)]
        omega_den = [np.zeros([self.dynamics[0].n_basis + 1, self.dynamics[0].n_basis + 1])
                        for s in range(self.n_modes)]
        sigma_num = [np.zeros([self.n_dim, self.n_dim]) for _s in range(self.n_modes)]
        sigma_den = [0 for _s in range(self.n_modes)]
        for k in range(len(data_set)):
            for _s in range(self.n_modes):
                omega_num[_s] += weights_num[k][_s]
                omega_den[_s] += weights_den[k][_s]
                sigma_num[_s] += cov_num[k][_s]
                sigma_den[_s] += cov_den[k][_s]
        for _s in range(self.n_modes):
            self.dynamics[_s].weights = np.dot(omega_num[_s], np.linalg.pinv(omega_den[_s]))
            self.sigma_set[_s] = sigma_num[_s] / sigma_den[_s]

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