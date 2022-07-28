import numpy as np

def normal_prob(y, mu, sigma, inv=False):
    '''
    Return the normal pdf with mean mu and covariance matrix sigma evaluated in y.
    If the inverse covariance matrix is passed, set inv=True.
    '''
    if np.isscalar(mu) and np.isscalar(sigma) and np.isscalar(y):
        if not inv:
            return 0.3989422804014327 / np.sqrt(sigma) * np.exp(-0.5 * (y - mu) * (y - mu) / sigma)
        else:
            return 0.3989422804014327 * np.sqrt(sigma) * np.exp(-0.5 * (y - mu) * (y - mu) * sigma)
    else:
        # Check number of dimensions
        if not (np.ndim(mu) == 1 and np.ndim(sigma) == 2 and np.ndim(y) == 1):
            raise ValueError('mu and y must be a 1d array and sigma a 2d array')
        k = np.shape(mu)[0]
        if not (np.shape(sigma) == (k, k)):
            raise ValueError('sigma must be a square matrix with number or rows equals to mu and y')
        if not inv:
            sigma_inv = np.linalg.pinv(sigma)
            det_sigma = np.linalg.det(sigma)
        else:
            sigma_inv = sigma
            det_sigma = 1.0 / np.linalg.det(sigma_inv)
        const = (0.3989422804014327 ** k) / np.sqrt(det_sigma)
        return const * np.exp(-0.5 * (y - mu) @ sigma_inv @ (y - mu))

def log_normal_prob(y, mu, sigma, inv=False):
    '''
    Return the normal pdf with mean mu and covariance matrix sigma evaluated in y.
    If the inverse covariance matrix is passed, set inv=True.
    '''
    if np.isscalar(mu) and np.isscalar(sigma) and np.isscalar(y):
        if not inv:
            return -0.9189385332046727 - 0.5 * np.log(sigma) - 0.5 * (y - mu) * (y - mu) / sigma
        else:
            return -0.9189385332046727 + 0.5 * np.log(sigma) - 0.5 * (y - mu) * (y - mu) * sigma
    else:
        # Check number of dimensions
        if not (np.ndim(mu) == 1 and np.ndim(sigma) == 2 and np.ndim(y) == 1):
            raise ValueError('mu and y must be a 1d array and sigma a 2d array')
        k = np.shape(mu)[0]
        if not (np.shape(sigma) == (k, k)):
            raise ValueError('sigma must be a square matrix with number or rows equals to mu and y')
        if not inv:
            sigma_inv = np.linalg.pinv(sigma)
            det_sigma = np.linalg.det(sigma)
        else:
            sigma_inv = sigma
            det_sigma = 1.0 / np.linalg.det(sigma)
        return - k * 0.9189385332046727 - 0.5 * np.log(det_sigma) - 0.5 * \
            (y - mu) @ sigma_inv @ (y - mu)

## ----------------------------------------------------------------------------------------- ##
##  WARNING: This are not actual normalization, the output will always sum to 1, and not     ##
##           necessarily have unitary norm.                                                  ##
## ----------------------------------------------------------------------------------------- ##

def normalize_vect(y):
    '''
    Return the normalize vector.
    '''
    return y / np.sum(y)

def normalize_rows(M):
    '''
    Normalize each row of the given matrix.
    '''
    return M / np.array([np.sum(M, 1)]).T

def normalize_mtrx(M):
    '''
    Normalize the matrix to sum to 1.
    '''
    return M / np.sum(M)

#──────────────────────#
# QUATERNION FUNCTIONS #
#──────────────────────#

# the quaternion is a 4-d array [w x y z]

def quaternion_exponential(q):
    if q.ndim == 1:
        q_out = np.block([
            np.exp(q[0]) * np.cos(np.linalg.norm(q[1:])),
            np.exp(q[0]) * np.sin(np.linalg.norm(q[1:])) / np.linalg.norm(q[1:]) * q[1:]
            ])
    else:
        T = q.shape[0]
        exp_q_scal = np.reshape(
                np.exp(q[:, 0]) * np.linalg.norm(q[:, 1:], axis=1),
                [T, 1]
                )
        exp_q_vec = np.reshape(np.exp(q[:,0]) * np.sin(np.linalg.norm(q[:, 1:], axis=1)) / np.linalg.norm(q[:, 1:], axis=1), [T, 1]) * q[:, 1:]
        q_out = np.block([
            exp_q_scal, exp_q_vec
            ])
    return q_out

def quaternion_product(q0, q1):
    if q0.ndim == 1 and q1.ndim == 1:
        q_out = np.block([
            q0[0] * q1[0] - np.dot(q0[1:], q1[1:]),
            q0[0] * q1[1:] + q1[0] * q0[1:] + np.cross(q0[1:], q1[1:])
            ])
    elif q0.ndim == 1 and q1.ndim == 2:
        T = q1.shape[0]
        q0 = np.tile(q0, (T, 1))
        q_prod_scal = np.reshape(
                q0[:, 0] * q1[:, 0] - np.sum(q0[:, 1:] * q1[:, 1:] , 1),
            [T, 1])
        q_prod_vec = \
            np.reshape(q0[:,0], [T, 1]) * q1[:,1:] + \
            np.reshape(q1[:,0], [T, 1]) * q0[:,1:] + \
            np.cross(q0[:,1:], q1[:,1:])
        q_out = np.block([q_prod_scal, q_prod_vec])
    elif q0.ndim == 2 and q1.ndim == 1:
        T = q0.shape[0]
        q1 = np.tile(q1, (T, 1))
        q_prod_scal = np.reshape(
                q0[:, 0] * q1[:, 0] - np.sum(q0[:, 1:] * q1[:, 1:] , 1),
            [T, 1])
        q_prod_vec = \
            np.reshape(q0[:,0], [T, 1]) * q1[:,1:] + \
            np.reshape(q1[:,0], [T, 1]) * q0[:,1:] + \
            np.cross(q0[:,1:], q1[:,1:])
        q_out = np.block([q_prod_scal, q_prod_vec])
    elif q0.ndim == 2 and q1.ndim == 2:
        T = q0.shape[0]
        q_prod_scal = np.reshape(
                q0[:, 0] * q1[:, 0] - np.sum(q0[:, 1:] * q1[:, 1:] , 1),
            [T, 1])
        q_prod_vec = \
            np.reshape(q0[:,0], [T, 1]) * q1[:,1:] + \
            np.reshape(q1[:,0], [T, 1]) * q0[:,1:] + \
            np.cross(q0[:,1:], q1[:,1:])
        q_out = np.block([q_prod_scal, q_prod_vec])
    else:
        ValueError("quaternion_product: arguments must be 1-D or 2-D arrays. If both are 2-D, they must have identical shape")
    return q_out

#───────#
# TESTS #
#───────#

q0 = np.random.rand(4)
q1 = np.random.rand(4)
q2 = np.random.rand(10, 4)
q3 = np.random.rand(10, 4)

#print(quaternion_product(q0, q1))
#print(quaternion_product(q0, q2))
print(quaternion_product(q0, q2)[2] - quaternion_product(q0, q2[2]))
#print(quaternion_product(q2, q1))
print(quaternion_product(q2, q1)[2] - quaternion_product(q2[2], q1))
#print(quaternion_product(q2, q3))
print(quaternion_product(q2, q3)[2] - quaternion_product(q2[2], q3[2]))
