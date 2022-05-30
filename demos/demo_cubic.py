import numpy as np
from nl_arhmm.dynamic import Cubic_Dynamic
from nl_arhmm.dynamic import GRBF_Dynamic
from nl_arhmm.arhmm import Cubic_ARHMM
from nl_arhmm.transition import Transition
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
dyn_st = Cubic_Dynamic(2)
dyn_ol = Cubic_Dynamic(2)

# Creating the data sample to infer the dynamics
data_in = []
data_out_stable = []
data_out_omega_lim = []
dt = 0.1
for _ in range(500):
    _rho = np.random.rand()
    _theta = np.random.rand() * 2.0 * np.pi
    _in = _rho * np.array([np.cos(_theta), np.sin(_theta)])
    data_in.append(_in)
    data_out_stable.append(_in + dt * vf_stable(_in))
    data_out_omega_lim.append(_in + dt * vf_omega_lim(_in))

dyn_st.learn_vector_field(data_in, data_out_stable)
dyn_ol.learn_vector_field(data_in, data_out_omega_lim)

# Creating the NL - ARHMM
model = Cubic_ARHMM(2, 2)
model.dynamics[0].weights = dyn_st.weights
model.dynamics[1].weights = dyn_ol.weights
model.sigma_set[0] = 0.1 * np.eye(2)
model.sigma_set[1] = 0.1 * np.eye(2)

trans = Transition(2, np.array([[0.95, 0.05], [0.05, 0.95]]))
model.transition = trans

T = 200
sigma = np.array([[1, 0],
                  [0, 1]])
state = []
mode_true = []
num_signal = 20
for _ in range(num_signal):
    _rho = np.random.rand()
    _theta = np.random.rand() * 2.0 * np.pi
    _in = _rho * np.array([np.cos(_theta), np.sin(_theta)])
    [_state, _mode_true] = model.simulate(_in, T)
    state.append(np.nan_to_num(_state))
    mode_true.append(_mode_true)

mode_inferred = model.viterbi(state[0])

model.initialize(state, use_pos=True)
model.em_algorithm(state)

mode_inferred_em = model.viterbi(state[0])

plt.figure()
plt.subplot(311)
plt.imshow(np.array([mode_true[0]]), aspect='auto')
plt.subplot(312)
plt.imshow(np.array([mode_inferred]), aspect='auto')
plt.subplot(313)
plt.imshow(np.array([mode_inferred_em]), aspect='auto')

plt.figure()
plt.subplot(211)
plt.plot(state[0][:, 0], 'r')
plt.xlim(0, len(state[0][:,0]) - 1)
plt.subplot(212)
plt.plot(state[0][:, 1], 'g')
plt.xlim(0, len(state[0][:,0]) - 1)

plt.figure()
plt.plot(state[0][:,0], state[0][:,1])

plt.show()