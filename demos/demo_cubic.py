# ========================== #
#    ===  CODE SETUP  ===    #
# ========================== #

import numpy as np

from dataset.panda import panda as panda_dataset
from dataset.jigsaw import jigsaw as jigsaw_dataset
from dataset.davinci_pr import home as davinci_dataset

from dataset.preprocessing import normalize_data

from nl_arhmm.arhmm import Linear_ARHMM
from nl_arhmm.arhmm import Quadratic_ARHMM
from nl_arhmm.arhmm import Cubic_ARHMM

# --- Plotting stuff --- #
from matplotlib import rc
import matplotlib.pyplot as plt

fs = 14
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],
    'size':fs})
rc('text', usetex=True)

import matplotlib as mpl
from matplotlib import colors
cmap = colors.ListedColormap(['tab:purple', 'yellow', 'tab:green', 'tab:red', 'tab:blue', 'tab:orange'])
bounds = [0, 1, 2, 3, 4, 5]
norm = colors.BoundaryNorm(bounds, cmap.N)

# ================================ #
#    ===  MODEL DEFINITION  ===    #
# ================================ #

dt = 1.0 / 20

def vf(x):
    return np.array([
        x[0] ** 3 + x[1] ** 2 * x[0] - x[0] - x[1],
        x[1] ** 3 + x[0] ** 2 * x[1] + x[0] - x[1],
    ])

def alpha_vf(x):
    return x + dt * vf(x)

def omega_vf(x):
    return x - dt * vf(x)

x_in = []
x_out_alpha = []
x_out_omega = []
for _ in range(100):
    _r = np.random.rand()
    _theta = np.random.rand() * 2 * np.pi
    _x = np.array([_r * np.cos(_theta), _r * np.sin(_theta)])
    x_in.append(_x)
    x_out_alpha.append(alpha_vf(_x))
    x_out_omega.append(omega_vf(_x))

model_true = Quadratic_ARHMM(2, 2)
model_true.dynamics[0].learn_vector_field(x_in, x_out_alpha)
model_true.dynamics[1].learn_vector_field(x_in, x_out_omega)
model_true.initial.density = np.array([0.5, 0.5])
model_true.initial.loginit = np.log(model_true.initial.density)
model_true.transition.trans_mtrx = np.array([[0.95, 0.05], [0.05, 0.95]])
model_true.transition.logtrans = np.log(model_true.transition.trans_mtrx)
err_std = 0.005
model_true.dynamics[0].covariance = err_std * np.eye(model_true.n_dim)
model_true.dynamics[1].covariance = err_std * np.eye(model_true.n_dim)
print(model_true.dynamics[0].covariance)
print(model_true.dynamics[1].covariance)
out_set = [model_true.simulate(np.random.rand(2)/1.5, 100) for _ in range(50)]
data_set = [out_set[_n][0] for _n in range(len(out_set))]
label_set = [out_set[_n][1] for _n in range(len(out_set))]

label_true = model_true.viterbi(data_set[0])

data_set_standard = normalize_data(data_set)

## Model testing

model_cub = Cubic_ARHMM(2, 2)
model_lin = Linear_ARHMM(2, 2)

model_cub.initialize(data_set_standard, use_pos=False, use_diff=True)
model_lin.initialize(data_set_standard, use_pos=False, use_diff=True)

model_cub.em_algorithm(data_set_standard)
model_lin.em_algorithm(data_set_standard)

print(model_cub.dynamics[0].covariance)
print(model_cub.dynamics[1].covariance)

label_cub = model_cub.viterbi(data_set_standard[0])
label_lin = model_lin.viterbi(data_set_standard[0])

fig = plt.figure()
ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5)
ax1.plot(data_set_standard[0][:, 0])
ax1.plot(data_set_standard[0][:, 1])
ax1.set_xlim(0, data_set_standard[0].shape[0])
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
ax2 = plt.subplot2grid((8, 1), (5, 0), rowspan=1)
ax2.imshow(np.array([label_set[0]]), aspect='auto', cmap=cmap, norm=norm, interpolation='None')
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
ax3 = plt.subplot2grid((8, 1), (6, 0), rowspan=1)
ax3.imshow(np.array([label_lin]), aspect='auto', cmap=cmap, norm=norm, interpolation='None')
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
ax4 = plt.subplot2grid((8, 1), (7, 0), rowspan=1)
ax4.imshow(np.array([label_cub]), aspect='auto', cmap=cmap, norm=norm, interpolation='None')
frame1 = plt.gca()
# frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
ax4.set_xlabel(r"$t$")
# plt.savefig("imgs/validation_test_errstd" + str(err_std) + "_test8.png", dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.show()
