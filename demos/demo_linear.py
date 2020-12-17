import numpy as np
from nl_arhmm.arhmm import Linear_ARHMM
from nl_arhmm.transition import Transition
from nl_arhmm.initial import Initial
import matplotlib.pyplot as plt

# ARHMM

# True model
A_counterclock = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2.0)
b_counterclock = np.array([0.5, 0.5])
A_clock = np.array([[1.0, 1.0], [-1.0, 1.0]]) / np.sqrt(2.0)
b_clock = np.array([-0.5, -0.5])

M_counterclock = np.zeros([2, 3])
M_counterclock[:, 0] = b_counterclock
M_counterclock[:, 1:] = A_counterclock
M_clock = np.zeros([2, 3])
M_clock[:, 0] = b_clock
M_clock[:, 1:] = A_clock
dyn_mtrxs = [M_counterclock, M_clock]
sigmas = []
for _n in range(4):
    rm = np.eye(2) * 0.1
    sigmas.append(rm)
model_true = Linear_ARHMM(2, 2, dyn_mtrxs, sigmas)
trans = Transition(2, np.array([[0.95, 0.05], [0.1, 0.9]]))
model_true.transition = trans
init = Initial(2, np.array([0.5, 0.5]))
model_true.initial = init

x_true = []
mode_true = []
num_signals = 20
for n in range(num_signals):
    [_x, _mode] = model_true.simulate(np.random.rand(2))
    x_true.append(_x)
    mode_true.append(_mode)

mode_infered_true = model_true.viterbi(x_true[0])

# Test model
dyn_mtrxs = [0.5 * np.random.rand(2,3), 0.5 * np.random.rand(2,3)]
sigmas = []
for _n in range(4):
    rm = np.eye(2)
    sigmas.append(rm)
model = Linear_ARHMM(2, 2, dyn_mtrxs, sigmas)
model.initialize(x_true, use_pos=False)
model.em_algorithm(x_true)
mode_infered = model.viterbi(x_true[0])

# ---  Printing  ---
print(" --  True Model  -- ")
print("Initial density")
print(model_true.initial.density)
print("Transition matrix")
print(model_true.transition.trans_mtrx)
print("Vector field")
print(model_true.dynamics[0].weights)
print(model_true.dynamics[1].weights)

print(" --  Inferred Model  -- ")
print("Initial density")
print(model.initial.density)
print("Transition matrix")
print(model.transition.trans_mtrx)
print("Vector field")
print(model.dynamics[0].weights)
print(model.dynamics[1].weights)

# ---  Plotting  ---
plt.figure()
plt.subplot(311)
plt.imshow(np.array([mode_true[0]]), aspect='auto')
plt.subplot(312)
plt.imshow(np.array([mode_infered_true]), aspect='auto')
plt.subplot(313)
plt.imshow(np.array([mode_infered]), aspect='auto')

plt.figure()
plt.subplot(211)
plt.plot(x_true[0][:, 0], 'r')
plt.xlim(0, len(x_true[0][:,0]) - 1)
plt.subplot(212)
plt.plot(x_true[0][:, 1], 'g')
plt.xlim(0, len(x_true[0][:,0]) - 1)

plt.figure()
plt.plot(x_true[0][:,0], x_true[0][:,1])

plt.show()