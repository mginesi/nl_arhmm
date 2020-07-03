import numpy as np
from dynamic import GRBF_Dynamic
from arhmm import GRBF_ARHMM
import matplotlib.pyplot as plt

## Real dynamics
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

## Setting of the real parameters
initial_distr = np.array([0.5, 0.5])
trans_mtrx = np.array([[0.8, 0.2],
                       [0.2, 0.8]])

# Generation of the dynamics
rho = np.linspace(0.0, 1.0, 3)
theta = np.linspace(0.0, 2.0 * np.pi, 9)
theta = theta[:-1]
centers = np.zeros([24, 2])
for _rho in range(3):
    for _theta in range(8):
        centers[8 * _rho + _theta, 0] = _rho * np.cos(_theta)
        centers[8 * _rho + _theta, 1] = _rho * np.sin(_theta)
dyn_st = GRBF_Dynamic(2, centers, 0.2 * np.ones(24))
dyn_ol = GRBF_Dynamic(2, centers, 0.2 * np.ones(24))

# Creating the data sample to infer the dynamics
data_in = []
data_out_stable = []
data_out_omega_lim = []
for _ in range(500):
    _rho = np.random.rand()
    _theta = np.random.rand() * 2.0 * np.pi
    _in = _rho * np.array([np.cos(_theta), np.sin(_theta)])
    data_in.append(_in)
    data_out_stable.append(vf_stable(_in))
    data_out_omega_lim.append(vf_omega_lim(_in))

dyn_st.learn_vector_field(data_in, data_out_stable)
dyn_ol.learn_vector_field(data_in, data_out_omega_lim)

# Creating the NL - ARHMM
model = GRBF_ARHMM(2, 2, [centers, centers], [0.2 * np.ones(24), 0.2 * np.ones(24)],
                 [dyn_st.weights, dyn_ol.weights], [0.2 * np.eye(2), 0.2 * np.eye(2)])

# Change the weights
# model.dynamics[0].weights *= 2 * np.random.rand()
# model.dynamics[1].weights *= 2 * np.random.rand()
model.dynamics[0].weights = np.random.rand(model.dynamics[0].weights.shape[0],
                                           model.dynamics[0].weights.shape[1])
model.dynamics[1].weights = np.random.rand(model.dynamics[1].weights.shape[0],
                                           model.dynamics[1].weights.shape[1])
model.dynamics[0].weights /= 10
model.dynamics[1].weights /= 10

T = 100
sigma = np.array([[1.2, 0.2],
                  [0.2, 1.2]])
state = []
mode_true = []
num_signal = 1
print(model.transition.trans_mtrx)
for _ in range(num_signal):
    _rho = np.random.rand()
    _theta = np.random.rand() * 2.0 * np.pi
    _in = _rho * np.array([np.cos(_theta), np.sin(_theta)])
    [_state, _mode_true] = model.simulate(_in, T, sigma)
    state.append(_state)
    mode_true.append(_mode_true)
model.em_algorithm(state)
print(model.transition.trans_mtrx)

mode_inferred = model.viterbi(state[-1])

plt.figure()
plt.subplot(211)
plt.imshow(np.array([mode_true[0]]), aspect='auto')
plt.subplot(212)
plt.imshow(np.array([mode_inferred]), aspect='auto')
plt.show()