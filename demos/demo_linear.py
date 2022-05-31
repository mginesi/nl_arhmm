import numpy as np
from nl_arhmm.arhmm import Linear_ARHMM
from nl_arhmm.arhmm import Quadratic_ARHMM
from nl_arhmm.transition import Transition
from nl_arhmm.initial import Initial
import matplotlib.pyplot as plt
from matplotlib import rc

fs = 14
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],
    'size':fs})
rc('text', usetex=True)

import matplotlib as mpl

import matplotlib as mpl
from matplotlib import colors
cmap = colors.ListedColormap(['tab:purple', 'yellow', 'tab:green', 'tab:red', 'tab:blue', 'tab:orange'])
bounds = [0, 1, 2, 3, 4, 5]
norm = colors.BoundaryNorm(bounds, cmap.N)

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
for _n in range(2):
    rm = np.eye(2) * 0.1
    sigmas.append(rm)
model_true = Linear_ARHMM(2, 2)
model_true.dynamics[0].weights = dyn_mtrxs[0]
model_true.dynamics[1].weights = dyn_mtrxs[1]
model_true.dynamics[0].covariance = sigmas[0]
model_true.dynamics[1].covariance = sigmas[1]
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
model = Linear_ARHMM(2, 2)
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
print(model_true.dynamics[0].covariance)
print(model_true.dynamics[1].covariance)

print(" --  Inferred Model  -- ")
print("Initial density")
print(model.initial.density)
print("Transition matrix")
print(model.transition.trans_mtrx)
print("Vector field")
print(model.dynamics[0].weights)
print(model.dynamics[1].weights)
print(model.dynamics[0].covariance)
print(model.dynamics[1].covariance)

# ---  Plotting  ---
plt.figure()
plt.subplot(311)
plt.imshow(np.array([mode_true[0]]), aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')#, cmap=cmap, norm=norm)
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
plt.subplot(312)
plt.imshow(np.array([mode_infered_true]), aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')#, cmap=cmap, norm=norm)
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
plt.subplot(313)
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.imshow(np.array([mode_infered]), aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')#, cmap=cmap, norm=norm)
plt.xlabel(r"$t$")

# plt.figure()
# plt.subplot(211)
# plt.plot(x_true[0][:, 0], 'r')
# frame1 = plt.gca()
# frame1.axes.xaxis.set_ticklabels([])
# plt.xlim(0, len(x_true[0][:,0]) - 1)
# plt.subplot(212)
# frame1 = plt.gca()
# frame1.axes.xaxis.set_ticklabels([])
# plt.plot(x_true[0][:, 1], 'g')
# plt.xlim(0, len(x_true[0][:,0]) - 1)

# plt.figure()
# plt.plot(x_true[0][:,0], x_true[0][:,1])

# Computing error
err_true = mode_true[0] - mode_infered_true
err_learned = mode_true[0] - mode_infered

print(" ---  Scores  --- ")
print(np.count_nonzero(err_true) / 101)
print(np.count_nonzero(err_learned) / 101)

plt.show()
