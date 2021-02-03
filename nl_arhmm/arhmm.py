import numpy as np
from scipy.special import logsumexp
import copy
from tqdm import tqdm

import multiprocessing

from nl_arhmm.initial import Initial
from nl_arhmm.transition import Transition
from nl_arhmm.utils import normal_prob, log_normal_prob, normalize_vect, normalize_rows, normalize_mtrx

class ARHMM(object):

    def __init__(self, n_dim, n_modes, dynamics, correction=1e-14):
        '''
        Class to implement Non-Linear Auto-Regressive Hidden Markov Models.
        '''
        self.n_dim = n_dim
        self.n_modes = n_modes
        self.correction = correction
        self.initial = Initial(self.n_modes)
        self.transition = Transition(self.n_modes)
        self.dynamics = dynamics
        self.sigma_set = [np.eye(self.n_dim) for _ in range(self.n_modes)]

    def give_log_likelihood(self, log_c_stream):
        '''
        Compute the likelihood of the data
        '''
        lkl = 0
        for _log_c in log_c_stream:
            lkl += np.sum(_log_c)
        return lkl

    def initialize(self, data_set, use_pos=True, use_diff=True, **kwargs):
        '''
        Estimate the initial guess of the parameters.
        '''

        if not isinstance(data_set, list):
            data_set = [data_set]

        # Preliminary data fit with k-Means
        from sklearn.cluster import KMeans
        times = np.array([len(_data) for _data in data_set])
        times = list(np.append(0, np.cumsum(times) - 1))

        x_diff = [np.diff(_data, axis=0) for _data in data_set]
        # Last two "velocities" are the same
        x_diff = [np.vstack((_x_diff, _x_diff[-1])) for _x_diff in x_diff]
        x_diff_all = np.vstack(x_diff)
        data_complete = np.hstack((np.vstack(data_set), x_diff_all))
        km = KMeans(self.n_modes)
        if use_pos and use_diff:
            data_to_cluster = data_complete
        elif use_pos and not use_diff:
            data_to_cluster = data_complete[:, :self.n_dim]
        elif not use_pos and use_diff:
            data_to_cluster = data_complete[:, self.n_dim:]
        km.fit(data_to_cluster)

        # First estimate of dynamics
        for _m in range(self.n_modes):
            input_set = data_complete[np.where(km.labels_ == _m)[0], self.n_dim:]
            output_set = input_set + data_complete[np.where(km.labels_ == _m)[0], :self.n_dim]
            self.dynamics[_m].learn_vector_field(input_set, output_set)
            self.sigma_set[_m] = self.dynamics[_m].estimate_cov_mtrx(input_set, output_set)

        # Estimate of initial density
        _init = np.zeros(self.n_modes)
        for _m in range(self.n_modes):
            _init[_m] = list(km.labels_[times]).count(_m)
        _init += self.correction # to avoid null components
        self.initial.density = normalize_vect(_init)
        self.initial.logint = np.log(self.initial.density)

        # Estimate of transition probability
        trans = [[km.labels_[t], km.labels_[t+1]] for t in range(len(km.labels_) - 1)]
        _T = np.zeros([self.n_modes, self.n_modes])
        for _m in range(self.n_modes):
            for _n in range(self.n_modes):
                _T[_m, _n] = trans.count([_m, _n])
        _T += self.correction # to avoid null components
        self.transition.trans_mtrx = normalize_rows(_T)
        self.transition.logtrans = np.log(self.transition.trans_mtrx)

    def gibbs_sampling(self, data_set, mode_set, i_max = 10):
        '''
        Perform a Gibbs Sampling like learning alternating Learning and segmentation.
        '''
        pool = multiprocessing.Pool()

        # Check if the inputs are lists
        if not(isinstance(data_set, list)):
            data_set = [data_set]
        if not(isinstance(mode_set, list)):
            mode_set = [mode_set]

        # Keep iterating learning and segmentation
        for _i in tqdm(range(i_max)):
            self.learn_parameters(data_set, mode_set)
            mode_set = pool.map(self.viterbi, data_set)

    def learn_parameters(self, data_set, mode_set):
        '''
        Learn the model parameters given both the observations and the hidden mode sequence.
        '''
        pool = multiprocessing.Pool()
        if not(isinstance(data_set, list)):
            data_set = [data_set]
        if not(isinstance(mode_set, list)):
            mode_set = [mode_set]

        n_data = len(data_set)

        # time series goes from 0 to T
        T_set = [len(data_set[_i]) - 1 for _i in range(len(data_set))]

        # --- Estimating the initial probability --- #
        in_count = np.asarray(pool.map(self._learn_count_initial, mode_set))
        coefs = np.reshape(np.asarray(T_set), [n_data, 1])
        self.initial.density = normalize_vect(np.sum(in_count * coefs, axis = 0) / sum(T_set) + self.correction)

        # --- Estimating the transition probability --- #
        trans_count = np.asanyarray(pool.map(self._learn_count_transition, mode_set)) # n_data x n_modes x n_modes
        coefs = np.reshape(np.asarray(T_set), [n_data, 1, 1])
        self.transition.trans_mtrx = normalize_rows(np.sum(trans_count * coefs, axis = 0) / sum(T_set) + self.correction)

        # --- Estimating the dynamic --- #
        # Done identically to maximize_emission in EM algorithm, with gamma being a dirac delta
        # distribution

        gamma_set = pool.map(self._learn_give_gamma_dirac, mode_set)
        self.maximize_emissions(gamma_set, data_set)

    def em_algorithm(self, data_set, tol = 0.05, max_iter = 10,verbose=True):
        '''
        Perform the EM algorithm.
        '''
        pool = multiprocessing.Pool()
        # Check if data_set is a single demonstration or a list
        if not(isinstance(data_set, list)):
            data_set = [data_set]

        # Perform EM algorithm
        count = 0

        # First iteration of the forward variable (needed to compute the scaling term c used to
        # compute the log likelihood)
        logprob_future_stream = pool.map(self.compute_log_probability, data_set)
        forward_stream = pool.map(self.compute_forward_var, logprob_future_stream)
        alpha_stream = [forward_stream[i][0] for i in range(len(forward_stream))]
        c_stream = [forward_stream[i][1] for i in range(len(forward_stream))]
        new_lh = self.give_log_likelihood(c_stream)
        print('Step 0: LH = ' + str(new_lh))
        convergence = False
        while not convergence:
            count += 1
            old_lh = copy.deepcopy(new_lh)

            # Compute the backward variables
            beta_stream = pool.map(self.compute_backward_var, zip(logprob_future_stream, c_stream))

            # Compute the marginals
            gamma_stream = pool.map(self.compute_gamma, zip(alpha_stream, beta_stream))
            xi_stream = pool.map(self.compute_xi, zip(alpha_stream, beta_stream, logprob_future_stream, c_stream))

            # Maximization Step
            self.maximize_initial(gamma_stream)
            self.maximize_transition(gamma_stream, xi_stream)
            self.maximize_emissions(gamma_stream, data_set)

            # Compute the forward variables and the rescaling terms
            logprob_future_stream = pool.map(self.compute_log_probability, data_set)
            forward_stream = pool.map(self.compute_forward_var, logprob_future_stream)
            alpha_stream = [forward_stream[i][0] for i in range(len(forward_stream))]
            c_stream = [forward_stream[i][1] for i in range(len(forward_stream))]
            new_lh = self.give_log_likelihood(c_stream)

            convergence = ((np.abs((new_lh - old_lh) / old_lh) < tol)) or (count > max_iter)
            if verbose:
                print('Step ' + str(count) + ': LH = ' + str(new_lh))

        return

    def give_prob_of_next_step(self, y0, y1, mode):
        mu = self.dynamics[mode].apply_vector_field(y0)
        return normal_prob(y1, mu, self.sigma_set[mode])

    def give_log_prob_of_next_step(self, y0, y1, mode):
        mu = self.dynamics[mode].apply_vector_field(y0)
        return log_normal_prob(y1, mu, self.sigma_set[mode])

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
            _y = self.dynamics[_mode].apply_vector_field(_y) + \
                np.dot(self.sigma_set[_mode], np.random.randn(self.n_dim))
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
            T_1[0][_s] = self.give_log_prob_of_next_step(data_stream[0], data_stream[1], _s) + \
                self.initial.loginit[_s]
            # T_2[0] = 0.0

        # Recursion over time
        # FIXME: is it possible to vectorize something?
        for _t in range(1, T):
            for _s in range(self.n_modes):
                to_maximize = T_1[_t - 1] + self.transition.logtrans[:, _s] + \
                    self.give_log_prob_of_next_step(data_stream[_t], data_stream[_t + 1], _s)
                _max = np.max(to_maximize)
                _argmax = np.where(to_maximize == _max)[0][0]
                T_1[_t, _s] = copy.deepcopy(_max)
                T_2[_t, _s] = copy.deepcopy(_argmax)

        # Extract the most probable sequence
        z = np.zeros(T)
        max_T = np.max(T_1[T - 1])
        argmax_T = np.where(T_1[T - 1] == max_T)[0][0]
        z[-1] = int(argmax_T)
        for _t in range(T - 1, 0, -1):
            z[_t - 1] = T_2[_t, int(z[_t])]
        return z

    # --------------------------------------------------------------------------------------- #
    #                              Expectation step functions                                 #
    # --------------------------------------------------------------------------------------- #

    def compute_log_probability(self, _data):
        logp_future = []
        T = len(_data) - 1
        _logp = np.zeros(self.n_modes)
        for _t in range(T):
            for _m in range(self.n_modes):
                _logp[_m] = log_normal_prob(_data[_t + 1],
                            self.dynamics[_m].apply_vector_field(_data[_t]),
                            self.sigma_set[_m])
            logp_future.append(copy.deepcopy(_logp))
        return logp_future

    def compute_forward_var(self, _logprob_data):
        '''
        Recursively compute the (logarithm of) scaled forward variables and the scaling factors
          alpha(z_t) = p (z_t | y_0, .. , y_{t+1}, Theta^{old})
          c_t = p (y_t | y_0, ..., y_{t-1}, Theta^{old})
        '''
        T = len(_logprob_data)
        log_alpha = np.zeros([T, self.n_modes])
        log_c_rescale = np.zeros(T)
        # Basis of recursion
        log_alpha[0] = _logprob_data[0] + self.initial.loginit
        log_c_rescale[0] = logsumexp(log_alpha[0])
        log_alpha[0] -= log_c_rescale[0]

        # Recursion
        for _t in range(1, T):
            log_alpha[_t] = _logprob_data[_t] + \
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
        _logprob_data = _in[0]
        _log_c_rescale = _in[1]
        # Initialization
        T = len(_logprob_data)
        log_beta = np.zeros([T, self.n_modes])

        # Recursion
        for _t in range(T - 2, -1, -1):
            log_beta[_t] = logsumexp(log_beta[_t + 1] + _logprob_data[_t + 1] + self.transition.logtrans,
                                     axis = 1) - _log_c_rescale[_t + 1]

        return log_beta

    def compute_gamma(self, _in):
        _log_alpha = _in[0]
        _log_beta = _in[1]
        return normalize_rows(np.exp(_log_alpha + _log_beta))

    def compute_xi(self, _in):
        log_alpha = _in[0]
        log_beta = _in[1]
        _logprob_future = _in[2]
        log_c_rescale = _in[3]
        T = np.shape(log_alpha)[0]
        _xi = np.zeros([T - 1, self.n_modes, self.n_modes])
        for _t in range(T - 1):
            _xi[_t] = (log_alpha[_t] + self.transition.logtrans.transpose()).transpose() + \
                _logprob_future[_t + 1] + log_beta[_t + 1] - log_c_rescale[_t + 1]
            _xi[_t] = normalize_mtrx(np.exp(_xi[_t]))
        return _xi

    # --------------------------------------------------------------------------------------- #
    #                              Maximization step functions                                #
    # --------------------------------------------------------------------------------------- #


    def maximize_initial(self, gamma_set):
        new_init = np.zeros(self.n_modes)
        K = len(gamma_set)
        for _, _gamma in enumerate(gamma_set):
            new_init += _gamma[0] / K
        new_init += self.correction
        self.initial.density = normalize_vect(new_init)
        self.initial.loginit = np.log(self.initial.density)

    def maximize_transition(self, gamma_set, xi_set):
        num = np.zeros([self.n_modes, self.n_modes])
        den = np.zeros(self.n_modes)
        for k in range(len(gamma_set)):
            num += np.sum(xi_set[k], axis=0)
            den += np.sum(gamma_set[k][:-1], axis=0)
        # The two corrections are applied following the sequent argument:
        # 1. Denominator should be non-zero in each component
        # 2. Transition matrix should ha no non-zero components so to not block transitions
        self.transition.trans_mtrx = normalize_rows(num / (np.reshape(den, [self.n_modes, 1]) + self.correction) + self.correction)
        self.transition.logtrans = np.log(self.transition.trans_mtrx)

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
            self.sigma_set[_s] = sigma_num[_s] / (sigma_den[_s] + self.correction) + self.correction * np.eye(self.n_dim)

    # --------------------------------------------------------------------------------------- #
    #              Subfunction for the "Learn with observed modes" part.                      #
    # --------------------------------------------------------------------------------------- #

    def _learn_count_initial(self, mode_seq):
        # --- Estimating the initial probability --- #
        in_count = np.zeros(self.n_modes)
        in_count[int(mode_seq[0])] += 1
        return in_count

    def _learn_count_transition(self, mode_seq):
        # --- Estimating the transition probability --- #
        # Count the total numer of transitions from one mode to the other
        tr_count = np.zeros([self.n_modes, self.n_modes])
        for _t in range(len(mode_seq) - 1):
            tr_count[int(mode_seq[_t]), int(mode_seq[_t + 1])] += 1
        return tr_count

    def _learn_give_gamma_dirac(self, mode_seq):
        gamma = np.zeros([len(mode_seq), self.n_modes])
        for _t in range(len(mode_seq)):
            gamma[_t][int(mode_seq[_t])] = 1
        gamma = normalize_rows(gamma + self.correction)
        return gamma

class GRBF_ARHMM(ARHMM):

    def __init__(self, n_dim, n_modes, dyn_center, dyn_widths, correction=1e-08):
        '''
        Class to implement Non-Linear Auto-Regressive Hidden Markov Models.
        '''
        from nl_arhmm.dynamic import GRBF_Dynamic

        self.n_dim = n_dim
        self.n_modes = n_modes
        self.initial = Initial(self.n_modes)
        self.transition = Transition(self.n_modes)
        self.sigma_set = [np.eye(self.n_dim) for _ in range(self.n_modes)]
        self.dynamics = []
        for _m in range(self.n_modes):
            self.dynamics.append(GRBF_Dynamic(self.n_dim, dyn_center[_m], dyn_widths[_m]))
        self.correction = correction
        self.model = ARHMM(self.n_dim, self.n_modes, self.dynamics, correction)

class Linear_ARHMM(ARHMM):

    def __init__(self, n_dim, n_modes, correction=1e-08):
        '''
        Class to implement Auto-Regressive Hidden Markov Models.
        '''
        from nl_arhmm.dynamic import Linear_Dynamic

        self.n_dim = n_dim
        self.n_modes = n_modes
        self.initial = Initial(self.n_modes)
        self.transition = Transition(self.n_modes)
        self.sigma_set = [np.eye(self.n_dim) for _ in range(self.n_modes)]
        self.dynamics = []
        for _m in range(self.n_modes):
            self.dynamics.append(Linear_Dynamic(self.n_dim))
        self.correction = correction
        self.model = ARHMM(self.n_dim, self.n_modes, self.dynamics, correction)

class Quadratic_ARHMM(ARHMM):

    def __init__(self, n_dim, n_modes, correction=1e-08):
        '''
        Class to implement Non-Linear Auto-Regressive Hidden Markov Models.
        '''
        from nl_arhmm.dynamic import Quadratic_Dynamic

        self.n_dim = n_dim
        self.n_modes = n_modes
        self.initial = Initial(self.n_modes)
        self.transition = Transition(self.n_modes)
        self.sigma_set = [np.eye(self.n_dim) for _ in range(self.n_modes)]
        self.dynamics = []
        for _m in range(self.n_modes):
            self.dynamics.append(Quadratic_Dynamic(self.n_dim))
        self.correction = correction
        self.model = ARHMM(self.n_dim, self.n_modes, self.dynamics, correction)

class Cubic_ARHMM(ARHMM):

    def __init__(self, n_dim, n_modes, correction=1e-08):
        '''
        Class to implement Non-Linear Auto-Regressive Hidden Markov Models.
        '''
        from nl_arhmm.dynamic import Cubic_Dynamic

        self.n_dim = n_dim
        self.n_modes = n_modes
        self.initial = Initial(self.n_modes)
        self.transition = Transition(self.n_modes)
        self.sigma_set = [np.eye(self.n_dim) for _ in range(self.n_modes)]
        self.dynamics = []
        for _m in range(self.n_modes):
            self.dynamics.append(Cubic_Dynamic(self.n_dim))
        self.correction = correction
        self.model = ARHMM(self.n_dim, self.n_modes, self.dynamics, correction)