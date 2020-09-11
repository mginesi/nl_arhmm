import numpy as np
import copy

class GRBF_Dynamic(object):

    def __init__(self, n_dim, centers, widths, weights=None):
        '''
        Class encoding the dynamic.
        '''
        self.n_dim = n_dim
        self.centers = centers
        self.n_basis = np.shape(centers)[0]
        self.widths = widths
        if weights is None:
            weights = np.zeros([self.n_dim, self.n_basis + 1])
        self.weights = weights

    def compute_phi_vect(self, x):
        '''
        Compute the vector phi(x) which components are
          phi_0 = 1
          phi_j = exp (- alpha_j || x - mu_j || ^ 2)
        '''
        phi = np.ones(self.n_basis + 1)
        phi[1:] = np.exp(-self.widths * np.linalg.norm(self.centers - x, axis=1) ** 2.0)
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
        return np.cov(np.transpose(err_set))

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

    def apply_vector_field(self, x):
        return np.dot(self.weights, self.compute_phi_vect(x))

class Linear_Dynamic(object):

    def __init__(self, n_dim, weights=None):
        '''
        Class encoding the dynamic.
        '''
        self.n_dim = copy.deepcopy(n_dim)
        self.n_basis = self.n_dim
        if weights is None:
            weights = np.zeros([self.n_dim, self.n_dim + 1])
        self.weights = copy.deepcopy(weights)

    def compute_phi_vect(self, x):
        '''
        Compute the vector phi(x) which components are
          phi_0 = 1
          phi_j = x_j
        '''
        phi = np.ones(self.n_dim + 1)
        phi[1:] = copy.deepcopy(x)
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
        return np.cov(np.transpose(err_set))

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

    def apply_vector_field(self, x):
        return np.dot(self.weights, self.compute_phi_vect(x))

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

    plt.show()