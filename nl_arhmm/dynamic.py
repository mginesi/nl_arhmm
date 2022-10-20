import numpy as np
from scipy.optimize import minimize as minimize_function
import copy
from nl_arhmm.utils import normal_prob, log_normal_prob
from nl_arhmm.utils import quaternion_product, quaternion_exponential
from nl_arhmm.utils import normalize_vect, normalize_rows
import multiprocessing

class GRBF_Dynamic(object):

    def __init__(self, n_dim, centers, widths):
        '''
        Class encoding the dynamic.
        '''
        self.n_dim = n_dim
        self.centers = centers
        self.widths = widths
        _phi = self.compute_phi_vect(np.zeros(n_dim))
        self.n_basis = len(_phi) - 1
        self.weights = np.zeros([self.n_dim, self.n_basis + 1])
        self.covariance = np.eye(self.n_dim)

    def compute_phi_vect(self, x):
        '''
        Compute the vector phi(x) which components are
          phi_0 = 1
          phi_j = exp (- alpha_j || x - mu_j || ^ 2)
        '''
        phi = np.ones(1)
        phi = np.append(phi, np.exp(-self.widths * np.linalg.norm(self.centers - x, axis=1) ** 2.0))
        return phi

    def estimate_cov_mtrx(self, input_set, output_set):
        '''
        Given a set of input vectors and a set of output vectors, return the covariance matrix.
        It is computed as the covariance of the errors when applying the vector field.
        '''
        # TODO: check if there is a better way!!
        pred_set = []
        for _, _in in enumerate(input_set):
            pred_set.append(self.apply_vector_field(_in))
        pred_set = np.asarray(pred_set)
        output_set = np.asarray(output_set)
        err_set = output_set - pred_set
        self.covariance = np.cov(err_set.T)

    def learn_vector_field(self, input_set, output_set, weights=None):
        '''
        Compute the set of weights given the input and output set.
        '''
        n_data = len(input_set)
        if weights is None:
            weights = np.ones(n_data)
        sqrt_weights = np.sqrt(np.asarray(weights))
        phi_mat = np.zeros([n_data, self.n_basis + 1])
        for _n in range(n_data):
            phi_mat[_n] = sqrt_weights[_n] * self.compute_phi_vect(input_set[_n])
        T = np.asarray(output_set) * np.reshape(sqrt_weights, [n_data, 1])
        self.weights = (np.linalg.pinv(phi_mat) @ T).T

    def maximize_emission_elements(self, in_arg):
        '''
        Perform the maximization step for the EM algorithm (for a single data stream)
        '''
        data = in_arg[0]
        gamma_s = in_arg[1]
        in_data = data[:-1]
        out_data = data[1:]
        T = np.shape(out_data)[0]
        phi_data = []
        expected_out_data = []
        for _, _in in enumerate(in_data):
            phi_data.append(self.compute_phi_vect(_in))
            expected_out_data.append(self.apply_vector_field(_in))
        expected_out_data = np.asarray(expected_out_data)

        ## Covariance matrix update
        err = np.reshape(out_data - expected_out_data, [T, self.n_dim, 1])
        err_t = np.reshape(out_data - expected_out_data, [T, 1, self.n_dim])
        cov_err = np.matmul(err, err_t)
        cov_num = np.sum(cov_err * np.reshape(gamma_s, [T, 1, 1]), 0)
        cov_den = np.sum(gamma_s)

        ## Weights update
        out_data_3d = np.reshape(np.asarray(out_data), [T, self.n_dim, 1])
        phi_in_row = np.reshape(np.asarray(phi_data), [T, 1, self.n_basis + 1])
        phi_in_column = np.reshape(np.asarray(phi_data), [T, self.n_basis + 1, 1])

        weights_num = np.sum(
                np.reshape(gamma_s, [T,1,1]) * np.matmul(out_data_3d, phi_in_row), 0)
        weights_den = np.sum(
                np.reshape(gamma_s, [T,1,1]) * np.matmul(phi_in_column, phi_in_row), 0)
        return [weights_num, weights_den, cov_num, cov_den]

    def maximize_emission(self, data_set, gamma_set, correction=1e-10):
        pool = multiprocessing.Pool()
        _out = pool.map(self.maximize_emission_elements, zip(data_set, gamma_set))
        # for _n in range(len(data_set)):
        #     _data = data_set[_n]
        #     _gamma = gamma_set[_n]
        #     _out = self.maximize_emission_elements([_data, _gamma])
        w_num = sum([_out[i][0] for i in range(len(_out))])
        w_den = sum([_out[i][1] for i in range(len(_out))])
        c_num = sum([_out[i][2] for i in range(len(_out))])
        c_den = sum([_out[i][3] for i in range(len(_out))])
        self.weights = w_num @ np.linalg.pinv(w_den)
        self.covariance = c_num / (c_den + correction) + correction * np.eye(self.n_dim)

    def give_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return normal_prob(y1, mu, self.covariance)

    def give_log_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return log_normal_prob(y1, mu, self.covariance)

    def apply_vector_field(self, x):
        return self.weights @ self.compute_phi_vect(x)

    def simulate_step(self, x):
        return self.apply_vector_field(x) + self.covariance @ np.random.randn(self.n_dim)

class Linear_Dynamic(object):

    def __init__(self, n_dim):
        '''
        Class encoding the dynamic.
        '''
        self.n_dim = copy.deepcopy(n_dim)
        self.n_basis = self.n_dim
        _phi = self.compute_phi_vect(np.zeros(n_dim))
        self.n_basis = len(_phi) - 1
        self.weights = np.zeros([self.n_dim, self.n_basis + 1])
        self.covariance = np.eye(self.n_dim)

    def compute_phi_vect(self, x):
        '''
        Compute the vector phi(x) which components are
          phi_0 = 1
          phi_j = x_j
        '''
        phi = np.ones(1)
        phi = np.append(phi, copy.deepcopy(x))
        return phi

    def estimate_cov_mtrx(self, input_set, output_set):
        '''
        Given a set of input vectors and a set of output vectors, return the covariance matrix.
        It is computed as the covariance of the errors when applying the vector field.
        '''
        # TODO: check if there is a better way!!
        pred_set = []
        for _, _in in enumerate(input_set):
            pred_set.append(self.apply_vector_field(_in))
        pred_set = np.asarray(pred_set)
        output_set = np.asarray(output_set)
        err_set = output_set - pred_set
        self.covariance = np.cov(err_set.T)

    def learn_vector_field(self, input_set, output_set, weights=None):
        '''
        Compute the set of weights given the input and output set.
        '''
        n_data = len(input_set)
        if weights is None:
            weights = np.ones(n_data)
        sqrt_weights = np.sqrt(np.asarray(weights))
        phi_mat = np.zeros([n_data, self.n_dim + 1])
        for _n in range(n_data):
            phi_mat[_n] = sqrt_weights[_n] * self.compute_phi_vect(input_set[_n])
        T = np.asarray(output_set) * np.reshape(sqrt_weights, [n_data, 1])
        self.weights = (np.linalg.pinv(phi_mat) @ T).T

    def maximize_emission_elements(self, in_arg):
        '''
        Perform the maximization step for the EM algorithm (for a single data stream)
        '''
        data = in_arg[0]
        gamma_s = in_arg[1]
        in_data = data[:-1]
        out_data = data[1:]
        T = np.shape(out_data)[0]
        phi_data = []
        expected_out_data = []
        for _, _in in enumerate(in_data):
            phi_data.append(self.compute_phi_vect(_in))
            expected_out_data.append(self.apply_vector_field(_in))
        expected_out_data = np.asarray(expected_out_data)

        ## Covariance matrix update
        err = np.reshape(out_data - expected_out_data, [T, self.n_dim, 1])
        err_t = np.reshape(out_data - expected_out_data, [T, 1, self.n_dim])
        cov_err = np.matmul(err, err_t)
        cov_num = np.sum(cov_err * np.reshape(gamma_s, [T, 1, 1]), 0)
        cov_den = np.sum(gamma_s)

        ## Weights update
        out_data_3d = np.reshape(np.asarray(out_data), [T, self.n_dim, 1])
        phi_in_row = np.reshape(np.asarray(phi_data), [T, 1, self.n_basis + 1])
        phi_in_column = np.reshape(np.asarray(phi_data), [T, self.n_basis + 1, 1])

        weights_num = np.sum(
                np.reshape(gamma_s, [T,1,1]) * np.matmul(out_data_3d, phi_in_row), 0)
        weights_den = np.sum(
                np.reshape(gamma_s, [T,1,1]) * np.matmul(phi_in_column, phi_in_row), 0)
        return [weights_num, weights_den, cov_num, cov_den]

    def maximize_emission(self, data_set, gamma_set, correction=1e-10):
        pool = multiprocessing.Pool()
        _out = pool.map(self.maximize_emission_elements, zip(data_set, gamma_set))
        # for _n in range(len(data_set)):
        #     _data = data_set[_n]
        #     _gamma = gamma_set[_n]
        #     _out = self.maximize_emission_elements([_data, _gamma])
        w_num = sum([_out[i][0] for i in range(len(_out))])
        w_den = sum([_out[i][1] for i in range(len(_out))])
        c_num = sum([_out[i][2] for i in range(len(_out))])
        c_den = sum([_out[i][3] for i in range(len(_out))])
        self.weights = w_num @ np.linalg.pinv(w_den)
        self.covariance = c_num / (c_den + correction) + correction * np.eye(self.n_dim)

    def give_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return normal_prob(y1, mu, self.covariance)

    def give_log_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return log_normal_prob(y1, mu, self.covariance)

    def apply_vector_field(self, x):
        return self.weights @ self.compute_phi_vect(x)

    def simulate_step(self, x):
        return self.apply_vector_field(x) + self.covariance @ np.random.randn(self.n_dim)

class Quadratic_Dynamic(object):

    def __init__(self, n_dim):
        self.n_dim = n_dim
        _phi = self.compute_phi_vect(np.zeros(n_dim))
        self.n_basis = len(_phi) - 1
        self.weights = np.zeros([self.n_dim, self.n_basis + 1])
        self.covariance = np.eye(self.n_dim)

    def compute_phi_vect(self, x):
        '''
        Compute the vector phi(x)
        phi_0(x) = 1
        phi_i(x) = x_i, i = 1, 2, ..., d
        phi_{d + d * (i - 1) + j} (x) = x_i x * j, i = 1, 2, ..., d, j = i, i+1, ..., d
        '''
        phi = np.ones(1 + self.n_dim)
        phi[0] = 1.0
        phi[1:] = x
        # TODO: vectorize me
        for _i in range(self.n_dim):
            phi = np.append(phi, x[_i] * x[_i:])
        return phi

    def estimate_cov_mtrx(self, input_set, output_set):
        '''
        Given a set of input vectors and a set of output vectors, return the covariance matrix.
        It is computed as the covariance of the errors when applying the vector field.
        '''
        # TODO: check if there is a better way!!
        pred_set = []
        for _, _in in enumerate(input_set):
            pred_set.append(self.apply_vector_field(_in))
        pred_set = np.asarray(pred_set)
        output_set = np.asarray(output_set)
        err_set = output_set - pred_set
        self.covariance = np.cov(err_set.T)

    def learn_vector_field(self, input_set, output_set, weights=None):
        '''
        Compute the set of weights given the input and output set.
        '''
        n_data = len(input_set)
        if weights is None:
            weights = np.ones(n_data)
        sqrt_weights = np.sqrt(np.asarray(weights))
        phi_mat = np.zeros([n_data, self.n_basis + 1])
        for _n in range(n_data):
            phi_mat[_n] = sqrt_weights[_n] * self.compute_phi_vect(input_set[_n])
        T = np.asarray(output_set) * np.reshape(sqrt_weights, [n_data, 1])
        self.weights = (np.linalg.pinv(phi_mat) @ T).T

    def maximize_emission_elements(self, in_arg):
        '''
        Perform the maximization step for the EM algorithm (for a single data stream)
        '''
        data = in_arg[0]
        gamma_s = in_arg[1]
        in_data = data[:-1]
        out_data = data[1:]
        T = np.shape(out_data)[0]
        phi_data = []
        expected_out_data = []
        for _, _in in enumerate(in_data):
            phi_data.append(self.compute_phi_vect(_in))
            expected_out_data.append(self.apply_vector_field(_in))
        expected_out_data = np.asarray(expected_out_data)

        ## Covariance matrix update
        err = np.reshape(out_data - expected_out_data, [T, self.n_dim, 1])
        err_t = np.reshape(out_data - expected_out_data, [T, 1, self.n_dim])
        cov_err = np.matmul(err, err_t)
        cov_num = np.sum(cov_err * np.reshape(gamma_s, [T, 1, 1]), 0)
        cov_den = np.sum(gamma_s)

        ## Weights update
        out_data_3d = np.reshape(np.asarray(out_data), [T, self.n_dim, 1])
        phi_in_row = np.reshape(np.asarray(phi_data), [T, 1, self.n_basis + 1])
        phi_in_column = np.reshape(np.asarray(phi_data), [T, self.n_basis + 1, 1])

        weights_num = np.sum(
                np.reshape(gamma_s, [T,1,1]) * np.matmul(out_data_3d, phi_in_row), 0)
        weights_den = np.sum(
                np.reshape(gamma_s, [T,1,1]) * np.matmul(phi_in_column, phi_in_row), 0)
        return [weights_num, weights_den, cov_num, cov_den]

    def maximize_emission(self, data_set, gamma_set, correction=1e-10):
        pool = multiprocessing.Pool()
        _out = pool.map(self.maximize_emission_elements, zip(data_set, gamma_set))
        # for _n in range(len(data_set)):
        #     _data = data_set[_n]
        #     _gamma = gamma_set[_n]
        #     _out = self.maximize_emission_elements([_data, _gamma])
        w_num = sum([_out[i][0] for i in range(len(_out))])
        w_den = sum([_out[i][1] for i in range(len(_out))])
        c_num = sum([_out[i][2] for i in range(len(_out))])
        c_den = sum([_out[i][3] for i in range(len(_out))])
        self.weights = w_num @ np.linalg.pinv(w_den)
        self.covariance = c_num / (c_den + correction) + correction * np.eye(self.n_dim)

    def give_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return normal_prob(y1, mu, self.covariance)

    def give_log_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return log_normal_prob(y1, mu, self.covariance)

    def apply_vector_field(self, x):
        return self.weights @ self.compute_phi_vect(x)

    def simulate_step(self, x):
        return self.apply_vector_field(x) + self.covariance @ np.random.randn(self.n_dim)

class Cubic_Dynamic(object):

    def __init__(self, n_dim):
        self.n_dim = n_dim
        _phi = self.compute_phi_vect(np.zeros(n_dim))
        self.n_basis = len(_phi) - 1
        self.weights = np.zeros([self.n_dim, self.n_basis + 1])
        self.covariance = np.eye(self.n_dim)

    def compute_phi_vect(self, x):
        '''
        Compute the vector phi(x)
        '''
        phi = np.ones(1 + self.n_dim)
        phi[0] = 1.0
        phi[1:] = x
        # TODO: vectorize me
        for _i in range(self.n_dim):
            phi = np.append(phi, x[_i] * x[_i:])
            for _j in range(_i, self.n_dim):
                phi = np.append(phi, x[_i] * x[_j] * x[_j:])
        return phi

    def estimate_cov_mtrx(self, input_set, output_set):
        '''
        Given a set of input vectors and a set of output vectors, return the covariance matrix.
        It is computed as the covariance of the errors when applying the vector field.
        '''
        # TODO: check if there is a better way!!
        pred_set = []
        for _, _in in enumerate(input_set):
            pred_set.append(self.apply_vector_field(_in))
        pred_set = np.asarray(pred_set)
        output_set = np.asarray(output_set)
        err_set = output_set - pred_set
        self.covariance = np.cov(err_set.T)

    def learn_vector_field(self, input_set, output_set, weights=None):
        '''
        Compute the set of weights given the input and output set.
        '''
        n_data = len(input_set)
        if weights is None:
            weights = np.ones(n_data)
        sqrt_weights = np.sqrt(np.asarray(weights))
        phi_mat = np.zeros([n_data, self.n_basis + 1])
        for _n in range(n_data):
            phi_mat[_n] = sqrt_weights[_n] * self.compute_phi_vect(input_set[_n])
        T = np.asarray(output_set) * np.reshape(sqrt_weights, [n_data, 1])
        self.weights = (np.linalg.pinv(phi_mat) @ T).T

    def maximize_emission_elements(self, in_arg):
        '''
        Perform the maximization step for the EM algorithm (for a single data stream)
        '''
        data = in_arg[0]
        gamma_s = in_arg[1]
        in_data = data[:-1]
        out_data = data[1:]
        T = np.shape(out_data)[0]
        phi_data = []
        expected_out_data = []
        for _, _in in enumerate(in_data):
            phi_data.append(self.compute_phi_vect(_in))
            expected_out_data.append(self.apply_vector_field(_in))
        expected_out_data = np.asarray(expected_out_data)

        ## Covariance matrix update
        err = np.reshape(out_data - expected_out_data, [T, self.n_dim, 1])
        err_t = np.reshape(out_data - expected_out_data, [T, 1, self.n_dim])
        cov_err = np.matmul(err, err_t)
        cov_num = np.sum(cov_err * np.reshape(gamma_s, [T, 1, 1]), 0)
        cov_den = np.sum(gamma_s)

        ## Weights update
        out_data_3d = np.reshape(np.asarray(out_data), [T, self.n_dim, 1])
        phi_in_row = np.reshape(np.asarray(phi_data), [T, 1, self.n_basis + 1])
        phi_in_column = np.reshape(np.asarray(phi_data), [T, self.n_basis + 1, 1])

        weights_num = np.sum(
                np.reshape(gamma_s, [T,1,1]) * np.matmul(out_data_3d, phi_in_row), 0)
        weights_den = np.sum(
                np.reshape(gamma_s, [T,1,1]) * np.matmul(phi_in_column, phi_in_row), 0)
        return [weights_num, weights_den, cov_num, cov_den]

    def maximize_emission(self, data_set, gamma_set, correction=1e-10):
        pool = multiprocessing.Pool()
        _out = pool.map(self.maximize_emission_elements, zip(data_set, gamma_set))
        # for _n in range(len(data_set)):
        #     _data = data_set[_n]
        #     _gamma = gamma_set[_n]
        #     _out = self.maximize_emission_elements([_data, _gamma])
        w_num = sum([_out[i][0] for i in range(len(_out))])
        w_den = sum([_out[i][1] for i in range(len(_out))])
        c_num = sum([_out[i][2] for i in range(len(_out))])
        c_den = sum([_out[i][3] for i in range(len(_out))])
        self.weights = w_num @ np.linalg.pinv(w_den)
        self.covariance = c_num / (c_den + correction) + correction * np.eye(self.n_dim)

    def give_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return normal_prob(y1, mu, self.covariance)

    def give_log_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return log_normal_prob(y1, mu, self.covariance)

    def apply_vector_field(self, x):
        return self.weights @ self.compute_phi_vect(x)

    def simulate_step(self, x):
        return self.apply_vector_field(x) + self.covariance @ np.random.randn(self.n_dim)

#─────────────────────────#
# UNIT QUATERNION DYNAMIC #
#─────────────────────────#

class Unit_Quaternion(object):

    def __init__(self, n_hands):
        self.n_hands = n_hands
        self.vf_coeff = [np.random.rand(3) for _ in range(self.n_hands)]
        self.vect_f = [quaternion_exponential(np.block([0, self.vf_coeff[_h]])) for _h in range(self.n_hands)]
        self.covariance_m = [np.eye(4) * 0.01 for _ in range(self.n_hands)]
        return

    # -- Functions to compute the gradient for the maximization step -- #
    # FIXME: in the learning, p changes but data doesn't, the function may be defined inside the learning method and be a function only on the "p" vector?

    def matrix_H(self, data, hand):
        T = len(data) - 1
        v_t = np.reshape(data[:-1, 0], [T, 1, 1]) # scalar component of quaternion data
        u_t = np.reshape(data[:-1, 1:], [T, 3, 1]) # vector component of quaternion data
        H_mtrx = np.zeros([T, 3, 4])
        # vectors and matrices reshaped to achieve a T x dim x 1 final result
        p_r = np.reshape(self.vf_coeff[hand], [1, 3, 1])
        p_norm = np.reshape(np.linalg.norm(self.vf_coeff[hand]), [1, 1, 1])
        sin_p = np.sin(p_norm)
        cos_p = np.cos(p_norm)
        # Skew symmetrix matrix
        skew_mtrx = np.zeros([T, 3, 3])
        skew_mtrx[:, 0, 1] = np.reshape(- u_t[:, 2], [T])
        skew_mtrx[:, 0, 2] = np.reshape(u_t[:, 1], [T])
        skew_mtrx[:, 1, 0] = np.reshape(u_t[:, 2], [T])
        skew_mtrx[:, 1, 2] = np.reshape(- u_t[:, 0], [T])
        skew_mtrx[:, 2, 0] = np.reshape(- u_t[:, 1], [T])
        skew_mtrx[:, 2, 1] = np.reshape(u_t[:, 0], [T])

        ## first column
        H_mtrx[:, :, 0] = np.reshape(
            - v_t * sin_p / p_norm * p_r +
            (p_norm * cos_p + sin_p / p_norm ** 3) * (np.transpose(p_r, [0, 2, 1]) @ u_t) * p_r +
            (- sin_p / p_norm) * u_t,
                [T, 3])

        ## last three columns
        H_mtrx[:, :, 1:] = \
            - sin_p / p_norm * p_r * np.transpose(u_t, [0, 2, 1]) + \
            v_t / p_norm / p_norm * (cos_p - sin_p / p_norm) * p_r * np.transpose(p_r, [0, 2, 1]) + \
            v_t * sin_p / p_norm * np.reshape(np.eye(3), [1, 3, 3]) + \
            (cos_p - sin_p / p_norm) / p_norm / p_norm * p_r @ np.transpose((np.cross(p_r, u_t, axis=1)), [0, 2, 1]) + \
            sin_p / p_norm * skew_mtrx

        return H_mtrx

    def gradient(self, data, gamma):
        T = len(data) - 1
        grad_list = [] # outputs: list of gradients (one for each hand)
        for hand in range(self.n_hands):
            # here we insert the "weight" given by gamma
            H = self.matrix_H(data, hand) * np.reshape(gamma, [T, 1, 1])
            err = data[1:] - quaternion_product(
                quaternion_exponential(
                    np.array([0, self.vf_coeff[hand][0], self.vf_coeff[hand][1], self.vf_coeff[hand][2]])),
                data[:-1])
            # The following gradient is (2-D) a 3 × 1 numpy array
            grad_list.append(-2 * np.sum(H @ np.reshape(np.linalg.pinv(self.covariance_m[hand]), [1, 4, 4]) @ np.reshape(err, [T, 4, 1]), axis=0))
        return grad_list

    def estimate_cov_mtrx(self, input_set, output_set, weights=None):
        return

    def learn_vector_field(self, input_set, output_set, weights=None):
        return

    def maximize_emission_elements(self, in_arg):
        return

    # ======================================================================== #
    #                         LIKELIHOOD MAXIMIZATION                          #

    def maximize_emission(self, data_set, gamma_set, correction=1e-10):
        '''
        # Function that has to be maximized
        def to_maximize_element(param, hand, data, gamma):
            exp_p = quaternion_product(
                quaternion_exponential(np.array([0, param[0], param[1], param[2]])),
                data[:-1])
            err = data[1:] - exp_p
            err = np.reshape(err, [T, 1, 4])
            err_t = np.transpose(err, [0, 2, 1])
            return np.sum(gamma * err @ self.covariance_m[hand] @ err_t, 0)
        # Inside this function we perform the gradient descent
        _out = pool.map(self.gradient, zip(data_set, gamma_set))
        '''

        # Covariance matrix
        # We define it now so that we can update the covariance matrices later without
        Sigma = np.zeros([4*self.n_hands, 4*self.n_hands])
        for _h in range(self.n_hands):
            Sigma[_h*4:(_h+1)*4, _h*4:(_h+1)*4] = self.covariance_m[_h]
        Sigma = np.reshape(Sigma, [1, 4*self.n_hands, 4*self.n_hands])
        # pool = multiprocessing.Pool()
        # def to_maximize(coeff):
        #     pool = multiprocessing.Pool()
        #     error_list = pool.map(self.to_maximize_single_data_stream, zip([self.vf_coeff for _ in range(len(data_set))], data_set, gamma_set, [Sigma for _ in range(len(data_set))]))
        #     return sum(error_list)
        def to_minimize(coeff):
            err_list = []
            for _n in range(len(data_set)):
                err_list.append(copy.deepcopy(self.to_minimize_single_data_stream([coeff, data_set[_n], gamma_set[_n], Sigma])))
            return sum(err_list)
        # sigma_out = pool.map(self.maximize_covariance_component, zip([self.vf_coeff for _ in range(len(data_set))], data_set, gamma_set))
        # sigma_num = sum([sigma_out[_y][0] for _y in range(len(data_set))])
        # sigma_den = sum([sigma_out[_y][1] for _y in range(len(data_set))])
        _sigma_num = []
        _sigma_den = []
        for _n in range(len(data_set)):
            [_sn, _sd] = self.maximize_covariance_component([self.vf_coeff, data_set[_n], gamma_set[_n]])
            _sigma_num.append(copy.deepcopy(_sn))
            _sigma_den.append(copy.deepcopy(_sd))
        sigma_num = sum(_sigma_num)
        sigma_den = sum(_sigma_den)

        # == We use the minimize function from the optim package in scipy to compute the coefficients of the vector field == #
        vf_coeff_list = [] # the vector field coefficient must be a list to be used in minimize_function
        for _h in range(self.n_hands):
            vf_coeff_list += [_coeff for _coeff in self.vf_coeff[_h]]
        _opt_out = minimize_function(to_minimize, copy.deepcopy(vf_coeff_list))
        _coeff = _opt_out.x

        for _h in range(self.n_hands):
            self.covariance_m[_h] = sigma_num[4*_h : 4*(_h+1)] / sigma_den
        return

    def to_minimize_single_data_stream(self, _args):
        coeff_list = _args[0]
        data = _args[1]
        T = data.shape[0]
        gamma = _args[2]
        Sigma = _args[3]
        # expected output
        coeff = []
        for _h in range(len(coeff_list) // 3):
            coeff.append(np.array(coeff_list[_h * 3 : (_h+1)*3]))
        exp_q = np.reshape(
            np.block([
                quaternion_product(
                    quaternion_exponential(np.array([0, coeff[_h][0], coeff[_h][1], coeff[_h][2]])),
                    data[:-1, 4*_h : 4*(_h + 1)]
                    )
                for _h in range(self.n_hands)]),
            [T-1, 1, 4*self.n_hands]
            )
        err_q = np.transpose(np.reshape(data[1:], [T-1, 4*self.n_hands,1]), [0,2,1]) - exp_q
        # error
        sum_error = np.sum(err_q @ Sigma @ np.transpose(err_q, [0,2,1]))
        return sum_error

    def maximize_covariance_component(self, _args):
        coeff = _args[0]
        data = _args[1]
        T = data.shape[0]
        gamma = _args[2]
        # expected output
        exp_q = np.reshape(
            np.block([
                quaternion_product(
                    quaternion_exponential(np.array([0, coeff[_h][0], coeff[_h][1], coeff[_h][2]])),
                    data[:-1, 4*_h : 4*(_h + 1)]
                    )
                for _h in range(self.n_hands)]),
            [T-1, 4*self.n_hands, 1]
            )
        err_q = np.transpose(np.reshape(data[1:], [T-1, 4*self.n_hands,1]), [0,2,1]) - exp_q
        sigma_num_tmp = np.sum(err_q @ np.transpose(err_q, [0, 2, 1]) * np.reshape(gamma, [T-1,1,1]), 0)
        # each hand is assumed to be independent to the others; thus we impose
        # the off-diagonal blocks to be zero
        sigma_num = np.zeros([4*self.n_hands, 4*self.n_hands])
        for _h in range(self.n_hands):
            sigma_num[4*_h:4*(_h+1), 4*_h:4*(_h+1)] = sigma_num_tmp[4*_h:4*(_h+1), 4*_h:4*(_h+1)]
        sigma_den = np.sum(gamma)
        return [sigma_num, sigma_den]

    # ======================================================================== #

    def give_prob_of_next_step(self, q_old, q_new):
        mu = self.apply_vector_field(q_old)
        Sigma = np.zeros([self.n_hands * 4, self.n_hans * 4])
        for _h in self.n_hands:
            Sigma[self.n_hands * 4 : (self.n_hands + 1) * 4, self.n_hands * 4 : (self.n_hands + 1) * 4] = self.covariance_m[_h]
        return normal_prob(q_new, mu, Sigma)

    def give_log_prob_of_next_step(self, q_old, q_new):
        mu = self.apply_vector_field(q_old)
        Sigma = np.zeros([self.n_hands * 4, self.n_hands * 4])
        for _h in range(self.n_hands):
            Sigma[_h * 4 : (_h + 1) * 4, _h * 4 : (_h + 1) * 4] = self.covariance_m[_h]
        return log_normal_prob(q_new, mu, Sigma)

    def apply_vector_field(self, q):
        list_out = []
        for _h in range(self.n_hands):
            list_out.append(
                normalize_vect(
                    quaternion_product(self.vect_f[_h], q[_h * 4: (_h + 1) * 4])
                ))
        out = np.block(list_out)
        return out

    def simulate_step(self, q):
        list_out = []
        for _h in range(self.n_hands):
            list_out.append(
                normalize_vect(
                    quaternion_product(self.vect_f[_h], q[_h * 4: (_h + 1) * 4]) + self.covariance_m[_h] @ np.random.randn(4)
                ))
        out = np.block(list_out)
        return out

#──────────────────────────────#
# DECOUPLED OBSERVED VARIABLES #
#──────────────────────────────#

class Multiple_Linear(object):

    def __init__(self, n_hand, n_dim):
        self.n_hand = n_hand
        self.n_dim = n_dim
        self.n_basis = (n_dim + 1) * n_hand
        self.weights = [np.random.rand(n_dim, n_dim + 1) for _ in range(self.n_hand)]
        self.covariance = [np.eye(n_dim) * 0.1 for _ in range(self.n_hand)]
        return

    def compute_phi_vect(self, x):
        phi = []
        _phi = np.ones(self.n_dim + 1)
        for _h in range(self.n_hand):
            _phi[:self.n_dim] = x[_h * self.n_dim : _h * self.n_dim + self.n_dim]
            phi.append(copy.deepcopy(_phi))
        return phi

    def estimate_cov_mtrx(self, input_set, output_set):
        pred_set = []
        for _, _in in enumerate(input_set):
            pred_set.append(self.apply_vector_field(_in))
        pred_set = np.asarray(pred_set)
        out_set = np.asarray(output_set)
        err_set = out_set - pred_set
        for _h in range(self.n_hand):
            self.covariance[_h] = np.cov((err_set[:, _h*self.n_dim:_h*self.n_dim + self.n_dim]).T)
        return

    def learn_vector_field(self, input_set, output_set, weights=None):
        n_data = len(input_set)
        if weights is None:
            weights = np.ones(n_data)
        sqrt_weights = np.sqrt(np.asarray(weights))
        for _h in range(self.n_hand):
            # Cartesian position
            phi_mat_pos = np.zeros([n_data, (self.n_dim + 1)])
            for _n in range(n_data):
                phi_mat_pos[_n] = sqrt_weights[_n] * self.compute_phi_vect(input_set[_n])[_h]
            T_hand = np.asarray(output_set)[:, _h*self.n_dim :_h*self.n_dim+self.n_dim] * np.reshape(sqrt_weights, [n_data , 1])
            self.weights[_h] = (np.linalg.pinv(phi_mat_pos) @ T_hand).T
        return

    def maximize_emission_elements(self, in_arg):
        data = in_arg[0]
        gamma_s = in_arg[1]
        in_data = data[:-1]
        out_data = data[1:]
        T = np.shape(out_data)[0]
        expected_out = []
        PHI = []
        for _, _in in enumerate(in_data):
            PHI += [self.compute_phi_vect(_in)]
            expected_out.append(self.apply_vector_field(_in))
        expected_out = np.asarray(expected_out)

        # for loop along number of end-effectors
        cov_num = []
        cov_den = []
        w_num = []
        w_den = []

        for _h in range(self.n_hand):

            _ex_out = expected_out[:, self.n_dim *_h : self.n_dim *_h + self.n_dim]
            _out = out_data[:, self.n_dim*_h : self.n_dim*_h + self.n_dim]
            _err = np.reshape(_out - _ex_out, [T, self.n_dim, 1])
            _err_t = np.reshape(_out - _ex_out, [T, 1, self.n_dim])
            _cov_err = np.matmul(_err, _err_t)
            _cov_num = np.sum(_cov_err * np.reshape(gamma_s, [T, 1, 1]), 0)
            _cov_den = np.sum(gamma_s)

            cov_num.append(_cov_num)
            cov_den.append(_cov_den)

            # Weights

            _out_3d = np.reshape(np.asarray(_out), [T, self.n_dim, 1])
            _phi = []
            for _n in range(len(PHI)):
                _phi.append(PHI[_n][_h])
            _phi_row = np.reshape(np.asarray(_phi), [T, 1, self.n_dim+1])
            _phi_column = np.reshape(np.asarray(_phi), [T, self.n_dim+1, 1])

            _w_num = np.sum(
                np.reshape(gamma_s, [T,1,1]) * np.matmul(_out_3d, _phi_row), 0
                    )
            _w_den = np.sum(
                np.reshape(gamma_s, [T,1,1]) * np.matmul(_phi_column, _phi_row), 0
                    )

            w_num.append(_w_num)
            w_den.append(_w_den)
        return [w_num, w_den, cov_num, cov_den]

    def maximize_emission(self, data_set, gamma_set, correction=1e-10):
        pool = multiprocessing.Pool()
        _out = pool.map(self.maximize_emission_elements, zip(data_set, gamma_set))
        for _h in range(self.n_hand):
            w_num = sum([_out[i][0][_h] for i in range(len(_out))])
            w_den = sum([_out[i][1][_h] for i in range(len(_out))])
            c_num = sum([_out[i][2][_h] for i in range(len(_out))])
            c_den = sum([_out[i][3][_h] for i in range(len(_out))])
            self.weights[_h] = w_num @ np.linalg.pinv(w_den)
            self.covariance[_h] = c_num / (c_den + correction) + correction * np.eye(self.n_dim)
        return

    def give_prob_of_netx_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        covariance = np.zeros([self.n_dim * self.n_hand, self.n_dim * self.n_hand])
        for _h in range(self.n_hand):
            covariance[(self.n_dim)*_h : (self.n_dim)*_h + self.n_dim, (self.n_dim)*_h : (self.n_dim)*_h + self.n_dim] = self.covariance[_h]
        return normal_prob(y1, mu, covariance)

    def give_log_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        covariance = np.zeros([self.n_dim * self.n_hand, self.n_dim * self.n_hand])
        for _h in range(self.n_hand):
            covariance[(self.n_dim)*_h : (self.n_dim)*_h + self.n_dim, (self.n_dim)*_h : (self.n_dim)*_h + self.n_dim] = self.covariance[_h]
        return log_normal_prob(y1, mu, covariance)

    def apply_vector_field(self, x):
        x_next = np.zeros(self.n_dim * self.n_hand)
        phi = self.compute_phi_vect(x)
        for _h in range(self.n_hand):
            x_next[_h * self.n_dim : _h * self.n_dim + self.n_dim] = self.weights[_h] @ phi[_h]
        return x_next

    def simulate_step(self, x):
        covariance = np.zeros([self.n_dim * self.n_hand, self.n_dim * self.n_hand])
        for _h in range(self.n_hand):
            covariance[(self.n_dim)*_h : (self.n_dim)*_h + self.n_dim, (self.n_dim)*_h : (self.n_dim)*_h + self.n_dim] = self.covariance[_h]
        return self.apply_vector_field(x) + covariance @ np.random.randn(self.n_dim)

class Linear_Hand_Quadratic_Gripper(object):

    def __init__(self, n_hand):
        self.n_hand = n_hand
        self.n_dim = 4 * self.n_hand
        self.n_basis = 9 * n_hand + 4 * n_hand
        self.weights_hand = [np.random.rand(3,4) for _ in range(self.n_hand)]
        self.weights_gripper = [np.random.rand(4) for _ in range(self.n_hand)]
        self.covariance_hand = [np.eye(3) * 0.1 for _ in range(self.n_hand)]
        self.covariance_gripper = [0.1 for _ in range(self.n_hand)]
        return

    def compute_phi_vect(self, x):
        return [[np.array([x[_h * 4], x[_h * 4 + 1], x[_h * 4 + 2], 1]), np.array([1, x[_h*4 + 3], x[_h*4 + 3] * x[_h*4 + 3]])] for _h in range(self.n_hand)]

    def estimate_cov_mtrx(self, input_set, output_set):
        pred_set = []
        for _, _in in enumerate(input_set):
            pred_set.append(self.apply_vector_field(_in))
        pred_set = np.asarray(pred_set)
        out_set = np.asarray(output_set)
        err_set = out_set - pred_set
        for _h in range(self.n_hand):
            self.covariance_hand[_h] = np.cov((err_set[:, _h*4:_h*4 + 3]).T)
            self.covariance_gripper[_h] = np.var(err_set[:, _h*4 + 3])
        return

    def learn_vector_field(self, input_set, output_set, weights=None):
        n_data = len(input_set)
        if weights is None:
            weights = np.ones(n_data)
        sqrt_weights = np.sqrt(np.asarray(weights))
        for _h in range(self.n_hand):
            # Cartesian position
            phi_mat_pos = np.zeros([n_data, 4])
            for _n in range(n_data):
                phi_mat_pos[_n] = sqrt_weights[_n] * self.compute_phi_vect(input_set[_n])[_h][0] # 0 is for the hand
            T_hand = np.asarray(output_set)[:, _h*4:_h*4+3] * np.reshape(sqrt_weights, [n_data , 1])
            self.weights_hand[_h] = (np.linalg.pinv(phi_mat_pos) @ T_hand).T
            # Gripper angle
            phi_mat_angle = np.zeros([n_data, 3])
            for _n in range(n_data):
                phi_mat_angle[_n] = sqrt_weights[_n] * self.compute_phi_vect(input_set[_n])[_h][1] # 0 is for the hand
            T_angle = (np.asarray(output_set)[:, _h*4+3]).reshape([n_data, 1]) * np.reshape(sqrt_weights, [n_data , 1])
            self.weights_gripper[_h] = (np.linalg.pinv(phi_mat_angle) @ T_angle).T
        return

    def maximize_emission_elements(self, in_arg):
        data = in_arg[0]
        gamma_s = in_arg[1]
        in_data = data[:-1]
        out_data = data[1:]
        T = np.shape(out_data)[0]
        expected_out = []
        PHI = []
        for _, _in in enumerate(in_data):
            PHI += [self.compute_phi_vect(_in)] # PHI is a n_data x n_hand x 2 list
            expected_out.append(self.apply_vector_field(_in))
        expected_out = np.asarray(expected_out)

        # for loop along number of end-effectors
        cov_num_h = []
        cov_den_h = []
        w_num_h = []
        w_den_h = []
        cov_num_g = []
        cov_den_g = []
        w_num_g = []
        w_den_g = []

        for _h in range(self.n_hand):

            #────────────────────#
            # Cartesian position #
            #────────────────────#

            # Covariance matrix

            _ex_out_h = expected_out[:, 4*_h : 4*_h + 3]
            _out_h = out_data[:, 4*_h : 4*_h + 3]
            _err_h = np.reshape(_out_h - _ex_out_h, [T, 3, 1])
            _err_h_t = np.reshape(_out_h - _ex_out_h, [T, 1, 3])
            _cov_err_h = np.matmul(_err_h, _err_h_t)
            _cov_num_h = np.sum(_cov_err_h * np.reshape(gamma_s, [T, 1, 1]), 0)
            _cov_den_h = np.sum(gamma_s)

            cov_num_h.append(_cov_num_h)
            cov_den_h.append(_cov_den_h)

            # Weights

            _out_h_3d = np.reshape(np.asarray(_out_h), [T, 3, 1])
            _phi_h = []
            for _n in range(len(PHI)):
                _phi_h.append(PHI[_n][_h][0]) # 0 is for the hand
            _phi_h_row = np.reshape(np.asarray(_phi_h), [T, 1, 4])
            _phi_h_column = np.reshape(np.asarray(_phi_h), [T, 4, 1])

            _w_num_h = np.sum(
                np.reshape(gamma_s, [T,1,1]) * np.matmul(_out_h_3d, _phi_h_row), 0
                    )
            _w_den_h = np.sum(
                np.reshape(gamma_s, [T,1,1]) * np.matmul(_phi_h_column, _phi_h_row), 0
                    )

            w_num_h.append(_w_num_h)
            w_den_h.append(_w_den_h)

            #───────────────#
            # Gripper angle #
            #───────────────#

            # Covariance matrix

            _ex_out_g = expected_out[:, 4*_h + 3]
            _out_g = out_data[:, 4*_h + 3]
            _err_g = np.reshape(_out_g - _ex_out_g, [T, 1, 1])
            _err_g_t = np.reshape(_out_g - _ex_out_g, [T, 1, 1])
            _cov_err_g = np.matmul(_err_g, _err_g_t)
            _cov_num_g = np.sum(_cov_err_g * np.reshape(gamma_s, [T, 1, 1]), 0)
            _cov_den_g = np.sum(gamma_s)

            cov_num_g.append(_cov_num_g)
            cov_den_g.append(_cov_den_g)

            # Weights

            _out_g_3d = np.reshape(np.asarray(_out_g), [T, 1, 1])
            _phi_g = []
            for _n in range(len(PHI)):
                _phi_g.append(PHI[_n][_h][1]) # 1 is for the gripper
            _phi_g_row = np.reshape(np.asarray(_phi_g), [T, 1, 3])
            _phi_g_column = np.reshape(np.asarray(_phi_g), [T, 3, 1])

            _w_num_g = np.sum(
                np.reshape(gamma_s, [T,1,1]) * np.matmul(_out_g_3d, _phi_g_row), 0
                    )
            _w_den_g = np.sum(
                np.reshape(gamma_s, [T,1,1]) * np.matmul(_phi_g_column, _phi_g_row), 0
                    )

            w_num_g.append(_w_num_g)
            w_den_g.append(_w_den_g)

        return [w_num_h, w_den_h, cov_num_h, cov_den_h, w_num_g, w_den_g, cov_num_g, cov_den_g]

    def maximize_emission(self, data_set, gamma_set, correction=1e-10):
        # TODO
        pool = multiprocessing.Pool()
        _out = pool.map(self.maximize_emission_elements, zip(data_set, gamma_set))
        for _h in range(self.n_hand):
            w_num_h = sum([_out[i][0][_h] for i in range(len(_out))])
            w_den_h = sum([_out[i][1][_h] for i in range(len(_out))])
            c_num_h = sum([_out[i][2][_h] for i in range(len(_out))])
            c_den_h = sum([_out[i][3][_h] for i in range(len(_out))])
            w_num_g = sum([_out[i][4][_h] for i in range(len(_out))])
            w_den_g = sum([_out[i][5][_h] for i in range(len(_out))])
            c_num_g = sum([_out[i][6][_h] for i in range(len(_out))])
            c_den_g = sum([_out[i][7][_h] for i in range(len(_out))])
            self.weights_hand[_h] = w_num_h @ np.linalg.pinv(w_den_h)
            self.covariance_hand[_h] = c_num_h / (c_den_h + correction) + correction * np.eye(3)
            self.weights_gripper[_h] = w_num_g @ np.linalg.pinv(w_den_g)
            self.covariance_gripper[_h] = (c_num_g / (c_den_g + correction) + correction)[0][0]
        return

    def give_prob_of_netx_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        covariance = np.zeros([4 * self.n_hand, 4*self.n_hand])
        for _h in range(self.n_hand):
            covariance[4*_h : 4*_h + 3, 4*_h : 4*_h + 3] = self.covariance_hand[_h]
            covariance[4*_h + 3, 4*_h + 3] = self.covariance_gripper[_h]
        return normal_prob(y1, mu, covariance)

    def give_log_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        covariance = np.zeros([4 * self.n_hand, 4*self.n_hand])
        for _h in range(self.n_hand):
            covariance[4*_h : 4*_h + 3, 4*_h : 4*_h + 3] = self.covariance_hand[_h]
            covariance[4*_h + 3, 4*_h + 3] = self.covariance_gripper[_h]
        return log_normal_prob(y1, mu, covariance)

    def apply_vector_field(self, x):
        x_next = np.zeros(4 * self.n_hand)
        phi = self.compute_phi_vect(x)
        for _h in range(self.n_hand):
            x_next[_h * 4 : _h * 4 + 3] = self.weights_hand[_h] @ phi[_h][0]
            x_next[_h * 4 + 3] = self.weights_gripper[_h] @ phi[_h][1]
        return x_next

    def simulate_step(self, x):
        covariance = np.zeros([4 * self.n_hand, 4*self.n_hand])
        for _h in range(self.n_hand):
            covariance[4*_h : 4*_h + 3, 4*_h : 4*_h + 3] = self.covariance_hand[_h]
            covariance[4*_h + 3, 4*_h + 3] = self.covariance_gripper[_h]
        return self.apply_vector_field(x) + covariance @ np.random.randn(self.n_dim)

class Pose_Linear(object):
    def __init__(self, n_hands):
        from nl_arhmm.dynamic import Linear_Dynamic, Unit_Quaternion
        self.n_hands = n_hands
        self.cart_dyn = [Linear_Dynamic(3) for _h in range(n_hands)]
        self.unit_quat = [Unit_Quaternion(1) for _h in range(n_hands)]
        return

    def learn_vector_field(self, _in_set, _out_set):
        return

    def estimate_cov_mtrx(self, _in_set, _out_set):
        return

    def simulate_step(self, y):
        y_out = np.zeros(7 * self.n_hands)
        for _h in range(self.n_hands):
            y_out[_h * 7 : _h * 7 + 3] = self.cart_dyn[_h].simulate_step(y[_h * 7 : _h * 7 + 3]) # h-th cartesian
            y_out[_h * 7 + 3 : _h * 7 + 7] = self.unit_quat[_h].simulate_step(y[_h * 7 + 3 : _h * 7 + 7]) # h-th quaternion
        return y_out

    def apply_vector_field(self, y):
        y_out = np.zeros(7 * self.n_hands)
        for _h in range(self.n_hands):
            y_out[_h * 7 : _h * 7 + 3] = self.cart_dyn[_h].apply_vector_field(y[_h * 7 : _h * 7 + 3]) # h-th cartesian
            y_out[_h * 7 + 3 : _h * 7 + 7] = self.unit_quat[_h].apply_vector_field(y[_h * 7 + 3 : _h * 7 + 7]) # h-th quaternion
        return y_out

    def give_log_prob_of_next_step(self, y_past, y_pres):
        logprob = 0
        for _h in range(self.n_hands):
            logprob += self.cart_dyn[_h].give_log_prob_of_next_step(y_past[_h * 7 : _h * 7 + 3], y_pres[_h * 7 : _h * 7 + 3]) # h-th cartesian
            logprob += self.unit_quat[_h].give_log_prob_of_next_step(y_past[_h * 7 + 3 : _h * 7 + 7], y_pres[_h * 7 + 3 : _h * 7 + 7]) # h-th quaternion
        return logprob

    def maximize_emission(self, data_set, gamma_set):
        for _h in range(self.n_hands):
            _data_set_cart = [_data[:, _h * 7 : _h * 7 + 3] for _data in data_set]
            _data_set_quat = [_data[:, _h * 7 + 3 : _h * 7 + 7] for _data in data_set]
            self.cart_dyn[_h].maximize_emission(_data_set_cart, gamma_set)
            self.unit_quat[_h].maximize_emission(_data_set_quat, gamma_set)
        return


#===============================================================================#
# The following code is a "template" on how to write a new ARHMM generalization #
#===============================================================================#
'''
class New_ARHMM(object):
    def __init__(self,):
        return

    #==========================================================================#
    # the next two functions are used only in the initialization of the AR-HMM #
    #==========================================================================#
    def learn_vector_field(self, _in_set, _out_set):
        return

    def estimate_cov_mtrx(self, _in_set, _out_set):
        return

    #=========================================================#
    # next functions are used to simulate and learn the model #
    #=========================================================#
    def simulate_step(self, y):
        return

    def apply_vector_field(self, y):
        return

    def give_log_prob_of_next_step(self, y0, y1):
        return

    def maximize_emission(self, data_set, gamma_set):
        return
'''

#───────────────────────────────#
# Methods to test the functions #
#───────────────────────────────#

if __name__ == "__main__":

    from dynamic import Pose_Linear
    from nl_arhmm.utils import normalize_rows

    # Setup dynamic
    n_hands = 2
    T = 20
    dyn = Pose_Linear(n_hands)

    # Testing the EM functions
    coeff = [np.random.rand(3) for _h in range(n_hands)]
    data = normalize_rows(np.random.rand(T, 7*n_hands))
    gamma = np.random.rand(T-1)
    print("test dynamic step")
    print(dyn.simulate_step(data[0]))
    print("test apply vector field")
    print(dyn.apply_vector_field(np.random.rand(7*n_hands)))
    print("test log probability")
    print(dyn.give_log_prob_of_next_step(data[0], data[1]))
    print("test maximize emission")
    print(dyn.maximize_emission([data], [gamma]))
