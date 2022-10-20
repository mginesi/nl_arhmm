import numpy as np
from nl_arhmm.arhmm import Pose_ARHMM, Unit_Quaternion_ARHMM
from nl_arhmm.transition import Transition
from nl_arhmm.initial import Initial
import matplotlib.pyplot as plt
from matplotlib import rc
from nl_arhmm.utils import normalize_vect
from nl_arhmm.utils import quaternion_exponential
import copy

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

model_true  = Pose_ARHMM(3, 2)
model_infer = Pose_ARHMM(3, 2)

# Model Simulation
n_data = 10
simulations = []
true_labels = []
for _n in range(n_data):
    [_sim, _mode] = model_true.simulate(normalize_vect(np.random.rand(4)), int(100 + np.floor(20 * np.random.rand()) - 10))
    simulations.append(copy.deepcopy(_sim))
    true_labels.append(copy.deepcopy(_mode))

# Model learning
model_infer.initialize(simulations)
model_infer.em_algorithm(simulations)

