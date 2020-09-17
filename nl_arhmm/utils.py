import numpy as np

def normal_prob(y, mu, sigma):
    '''
    Return the normal pdf with mean mu and variance sigma evaluated in y.
    '''
    if np.isscalar(mu) and np.isscalar(sigma) and np.isscalar(y):
        return 0.3989422804014327 / sigma * np.exp(-0.5 * (y - mu) * (y - mu) / sigma / sigma)
    else:
        # Check number of dimensions
        if not (np.ndim(mu) == 1 and np.ndim(sigma) == 2 and np.ndim(y) == 1):
            raise ValueError('mu and y must be a 1d array and sigma a 2d array')
        k = np.shape(mu)[0]
        if not (np.shape(sigma) == (k, k)):
            raise ValueError('sigma must be a square matrix with number or rows equals to mu and y')
        sigma_inv = np.linalg.pinv(sigma)
        det_sigma = np.linalg.det(sigma)
        const = (0.3989422804014327 ** k) / np.sqrt(det_sigma)
        return const * np.exp(-0.5 * np.dot(y - mu, np.dot(sigma_inv, y - mu)))

def log_normal_prob(y, mu, sigma):
    '''
    Return the normal pdf with mean mu and variance sigma evaluated in y.
    '''
    if np.isscalar(mu) and np.isscalar(sigma) and np.isscalar(y):
        return -0.918938533204673 - 0.5 * np.log(sigma) - 0.5 * (y - mu) * (y - mu) / sigma
    else:
        # Check number of dimensions
        if not (np.ndim(mu) == 1 and np.ndim(sigma) == 2 and np.ndim(y) == 1):
            raise ValueError('mu and y must be a 1d array and sigma a 2d array')
        k = np.shape(mu)[0]
        if not (np.shape(sigma) == (k, k)):
            raise ValueError('sigma must be a square matrix with number or rows equals to mu and y')
        sigma_inv = np.linalg.pinv(sigma)
        det_sigma = np.linalg.det(sigma)
        return - k * 0.918938533204673 - 0.5 * np.log(det_sigma) - 0.5 * \
            np.dot(y - mu, np.dot(sigma_inv, y - mu))

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
    return (M.transpose() / np.sum(M, 1)).transpose()

def normalize_mtrx(M):
    '''
    Normalize the matrix to sum to 1.
    '''
    return M / np.sum(M)