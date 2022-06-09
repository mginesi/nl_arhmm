import numpy as np
import copy
from nl_arhmm.utils import normal_prob, log_normal_prob
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
        self.covariance = np.cov(np.transpose(err_set))

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
        self.weights = np.transpose(np.dot(np.linalg.pinv(phi_mat), T))

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
        self.weights = np.dot(w_num, np.linalg.pinv(w_den))
        self.covariance = c_num / (c_den + correction) + correction * np.eye(self.n_dim)

    def give_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return normal_prob(y1, mu, self.covariance)

    def give_log_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return log_normal_prob(y1, mu, self.covariance)

    def apply_vector_field(self, x):
        return np.dot(self.weights, self.compute_phi_vect(x))

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
        self.covariance = np.cov(np.transpose(err_set))

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
        self.weights = np.transpose(np.dot(np.linalg.pinv(phi_mat), T))

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
        self.weights = np.dot(w_num, np.linalg.pinv(w_den))
        self.covariance = c_num / (c_den + correction) + correction * np.eye(self.n_dim)

    def give_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return normal_prob(y1, mu, self.covariance)

    def give_log_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return log_normal_prob(y1, mu, self.covariance)

    def apply_vector_field(self, x):
        return np.dot(self.weights, self.compute_phi_vect(x))

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
        self.covariance = np.cov(np.transpose(err_set))

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
        self.weights = np.transpose(np.dot(np.linalg.pinv(phi_mat), T))

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
        self.weights = np.dot(w_num, np.linalg.pinv(w_den))
        self.covariance = c_num / (c_den + correction) + correction * np.eye(self.n_dim)

    def give_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return normal_prob(y1, mu, self.covariance)

    def give_log_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return log_normal_prob(y1, mu, self.covariance)

    def apply_vector_field(self, x):
        return np.dot(self.weights, self.compute_phi_vect(x))

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
        self.covariance = np.cov(np.transpose(err_set))

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
        self.weights = np.transpose(np.dot(np.linalg.pinv(phi_mat), T))

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
        self.weights = np.dot(w_num, np.linalg.pinv(w_den))
        self.covariance = c_num / (c_den + correction) + correction * np.eye(self.n_dim)

    def give_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return normal_prob(y1, mu, self.covariance)

    def give_log_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return log_normal_prob(y1, mu, self.covariance)

    def apply_vector_field(self, x):
        return np.dot(self.weights, self.compute_phi_vect(x))

#──────────────────────────────#
# DECOUPLED OBSERVED VARIABLES #
#──────────────────────────────#

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
        return [[np.array([x[_h * 4 : _h*4 + 3]]), x[_h*4 + 3]] for _h in range(self.n_hand)]

    def estimate_cov_mtrx(self, input_set, output_set):
        pred_set = []
        for _, _in in enumerate(input_set):
            pred_set.append(self.apply_vector_field(_in))
        pred_set = np.asarray(pred_set)
        out_set = np.asarray(output_set)
        err_set = out_set - pred_set
        for _h in range(self.n_hand):
            self.covariance_hand[_h] = np.cov(np.transpose(err_set[_h*4:_h*4 + 3,:]))
            self.covariance_gripper[_h] = np.var(err_set[_h*4 + 3])
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
                phi_mat[_n] = sqrt_weights[_n] * self.compute_phi_vect(input_set[_n])[_h][0] # 0 is for the hand
            T_hand = np.asarray(output_set)[:, _h*4:_h*4+3] * np.reshape(sqrt_weights, [n_data , 1])
            self.weights_hand[_h] = np.transpose(np.dot(np.linalg.pinv(phi_mat), T_hand))
            # Gripper angle
            phi_mat_angle = np.zeros([n_data, 1])
            for _n in range(n_data):
                phi_mat[_n] = sqrt_weights[_n] * self.compute_phi_vect(input_set[_n])[_h][1] # 0 is for the hand
            T_angle = np.asarray(output_set)[:, _h*4+3] * np.reshape(sqrt_weights, [n_data , 1])
            self.weights_gripper[_h] = np.transpose(np.dot(np.linalg.pinv(phi_mat), T_angle))
        return

    def maximize_emission_elements(self, in_arg):
        # TODO
        data = in_arg[0]
        gamma_s = in_arg[1]
        in_data = data[:-1]
        out_data = data[1:]
        expected_out = []
        for _, _in in enumerate(in_data):
            PHI = self.compute_phi_vect(_in)
            expected_out.append(self.apply_vector_field(_in))
        expected_out = np.asarray(expected_out)

        # for loop along number of end-effectors
        for _h in range(self.n_hand):
        return

    def maximize_emission(self, data_set, gamma_set, correction=1e-10):
        # TODO
        return

    def give_prob_of_netx_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return normal_prob(y1, mu, self.covariance)

    def give_log_prob_of_next_step(self, y0, y1):
        mu = self.apply_vector_field(y0)
        return log_normal_prob(y1, mu, self.covariance)

    def apply_vector_field(self, x):
        x_next = np.zeros(4 * self.n_hand)
        phi = self.compute_phi_vect(x)
        for _h in range(self.n_hand):
            x_next[_h * 4 : _h * 4 + 3] = np.dot(self.weights_hand[_h], phi[_h][0])
            x_next[_h * 4 + 3] = np.dot(self.weights_gripper[_h], phi[_h][1])
        return x_next

# ------------------------------------------------------------------------------------------- #
#                                           Demo                                              #
# ------------------------------------------------------------------------------------------- #

if __name__ == "__main__":

    # GRBF

    from dynamic import GRBF_Dynamic
    import matplotlib.pyplot as plt
    rho = np.linspace(0.0, 1.0, 3)
    theta = np.linspace(0.0, 2.0 * np.pi, 9)
    theta = theta[:-1]
    centers = np.zeros([24, 2])
    for _rho in range(3):
        for _theta in range(8):
            centers[8 * _rho + _theta, 0] = _rho * np.cos(_theta)
            centers[8 * _rho + _theta, 1] = _rho * np.sin(_theta)
    dyn = GRBF_Dynamic(2, centers, 0.2 * np.ones(24))

    def vf_stable(x):
        x1 = x[0]
        x2 = x[1]
        f1 = x1 ** 3 + x2 ** 2 * x1 - x1 - x2
        f2 = x2 ** 3 + x1 ** 2 * x2 + x1 - x2
        return np.array([f1, f2])

    def vf_omega_lim(x):
        x1 = x[0]
        x2 = x[1]
        f1 = x1 ** 3 + x2 ** 2 * x1 - x1 - x2
        f2 = x2 ** 3 + x1 ** 2 * x2 + x1 - x2
        return - np.array([f1, f2])

    # Creation of the input and output set (Euler scheme used)
    n_sample = 500
    dt = 0.01
    input_set = []
    output_set = []

    # Stable data
    for _ in range(n_sample):
        _rho = np.random.rand()
        _theta = np.random.rand() * 2.0 * np.pi
        _in = _rho * np.array([np.cos(_theta), np.sin(_theta)])
        _out = _in + dt * vf_stable(_in)
        input_set.append(_in)
        output_set.append(_out)

    # Omega limit data
    for _ in range(n_sample):
        _rho = np.random.rand()
        _theta = np.random.rand() * 2.0 * np.pi
        _in = _rho * np.array([np.cos(_theta), np.sin(_theta)])
        _out = _in + dt * vf_omega_lim(_in)
        input_set.append(_in)
        output_set.append(_out)

    # Three set of weights
    w_stable = np.ones(2*n_sample)
    w_stable[n_sample:] = 0.0
    w_omega_lim = np.ones(2*n_sample)
    w_omega_lim[:n_sample] = 0.0
    w_mix = np.ones(2*n_sample)

    # Stable
    # Learning of the vector field
    dyn.learn_vector_field(input_set, output_set, w_stable)

    # Simulation of the learned v.f.
    rho = np.random.rand() * 0.5 + 0.5
    theta = np.random.rand() * 2.0 * np.pi
    _x = rho * np.array([np.cos(theta), np.sin(theta)])
    _x_t = rho * np.array([np.cos(theta), np.sin(theta)])
    x = [_x]
    x_true = [_x_t]
    for t in range(1000):
        _x = dyn.apply_vector_field(_x)
        _x_t = _x_t + dt * vf_stable(_x_t)
        x.append(_x)
        x_true.append(_x_t)
    x = np.asarray(x)
    x_true = np.asarray(x_true)
    plt.figure()
    plt.plot(x[:, 0], x[:, 1], 'r', label='learned')
    plt.plot(x_true[:, 0], x_true[:, 1], '--b', label='true')
    plt.plot(x[0, 0], x[0, 1], 'ok', label='x0')
    plt.plot(x[-1][0], x[-1][1], 'xk', label='xT')
    plt.legend(loc='best')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.title('GRBF - Only stable')

    # Omega lim
    # Learning of the vector field
    dyn.learn_vector_field(input_set, output_set, w_omega_lim)

    # Simulation of the learned v.f.
    rho = np.random.rand() * 0.5 + 0.5
    theta = np.random.rand() * 2.0 * np.pi
    _x = rho * np.array([np.cos(theta), np.sin(theta)])
    _x_t = rho * np.array([np.cos(theta), np.sin(theta)])
    x = [_x]
    x_true = [_x_t]
    for t in range(1000):
        _x = dyn.apply_vector_field(_x)
        _x_t = _x_t + dt * vf_omega_lim(_x_t)
        x.append(_x)
        x_true.append(_x_t)
    x = np.asarray(x)
    x_true = np.asarray(x_true)
    plt.figure()
    plt.plot(x[:, 0], x[:, 1], 'r', label='learned')
    plt.plot(x_true[:, 0], x_true[:, 1], '--b', label='true')
    plt.plot(x[0, 0], x[0, 1], 'ok', label='x0')
    plt.plot(x[-1][0], x[-1][1], 'xk', label='xT')
    plt.legend(loc='best')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.title('GRBF - Only omega lim')

    # Mixed
    # Learning of the vector field
    dyn.learn_vector_field(input_set, output_set, w_mix)

    # Simulation of the learned v.f.
    rho = np.random.rand() * 0.5 + 0.5
    theta = np.random.rand() * 2.0 * np.pi
    _x = rho * np.array([np.cos(theta), np.sin(theta)])
    _x_t = rho * np.array([np.cos(theta), np.sin(theta)])
    x = [_x]
    for t in range(1000):
        _x = dyn.apply_vector_field(_x)
        _x_t = _x_t + dt * vf_omega_lim(_x_t)
        x.append(_x)
    x = np.asarray(x)
    plt.figure()
    plt.plot(x[:, 0], x[:, 1], 'r', label='learned')
    plt.plot(x[0, 0], x[0, 1], 'ok', label='x0')
    plt.plot(x[-1][0], x[-1][1], 'xk', label='xT')
    plt.legend(loc='best')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.title('GRBF - Mixed')

    # Linear
    
    from dynamic import Linear_Dynamic

    A_true = np.random.rand(2, 2) / 10.0 # true v.f.
    b_true = np.random.rand(2) / 10.0
    n_sample = 1000
    in_set = []
    out_set = []
    for _n in range(n_sample):
        in_set.append(np.random.rand(2))
        out_set.append(np.dot(A_true, in_set[-1]) + b_true)

    dyn = Linear_Dynamic(2)
    dyn.learn_vector_field(in_set, out_set)

    T = 100
    x0 = np.random.rand(2)
    x_demo = np.zeros([T, 2])
    x_demo[0] = x0
    x_demo_true = copy.deepcopy(x_demo)
    for _t in range(T-1):
        x_demo_true[_t+1] = np.dot(A_true, x_demo_true[_t]) + b_true
        x_demo[_t + 1] = dyn.apply_vector_field(x_demo[_t])

    plt.figure()
    plt.plot(x_demo[:, 0], x_demo[:, 1], '-r', label='Inferred')
    plt.plot(x_demo_true[:, 0], x_demo_true[:, 1], '--b', label='True')
    plt.legend(loc='best')

    # Quadratic basis functions
    from dynamic import Quadratic_Dynamic
    dyn = Quadratic_Dynamic(2)
    x_0 = np.array([0.0, 0.9])
    _x = np.array([x_0])
    for _ in range(1000):
        _x = np.append(_x, np.array([_x[-1] + 0.01 * vf_stable(_x[-1]) ]), axis=0)
    dyn.learn_vector_field(_x[:-1], _x[1:])
    x = np.array([x_0])
    for _t in range(1000):
        x = np.append(x, np.array([dyn.apply_vector_field(x[-1])]), axis=0)
    dyn_l = Linear_Dynamic(2)
    dyn_l.learn_vector_field(_x[:-1], _x[1:])
    x_l = np.array([x_0])
    for _t in range(1000):
        x_l = np.append(x_l, np.array([dyn_l.apply_vector_field(x_l[-1])]), axis=0)
    from dynamic import Cubic_Dynamic
    q_dyn = Cubic_Dynamic(2)
    q_dyn.learn_vector_field(_x[:-1], _x[1:])
    x_q = np.array([x_0])
    for _t in range(1000):
        x_q = np.append(x_q, np.array([q_dyn.apply_vector_field(x_q[-1])]), axis=0)
    plt.figure()
    plt.plot(_x[:, 0], _x[:, 1], '-r', label='true')
    plt.plot(x[:, 0], x[:, 1], '--b', label='quad')
    plt.plot(x_l[:, 0], x_l[:, 1], ':g', label='linear')
    plt.plot(x_q[:, 0], x_q[:, 1], '-.', color='purple', label='cubic')
    plt.legend(loc='best')
    plt.title('Quadratic VS Linear')

    plt.show()
