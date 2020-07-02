import numpy as np
from nl_arhmm import Linear_ARHMM

import matplotlib.pyplot as plt

# Parameters of the true model

trans_mtrx = np.array([[0.8, 0.2], [0.2, 0.8]])
initial_distr = np.array([0.5, 0.5])
A_counterclock = np.array([[0.0, 1.0, -1.0], [0.0, 1.0, 1.0]]) / np.sqrt(2.0)
A_clock = np.array([[0.0, 1.0, 1.0], [0.0, -1.0, 1.0]]) / np.sqrt(2.0)
dyn_mtrxs = [A_counterclock, A_clock]
sigmas = [0.2*np.eye(2), 0.2*np.eye(2)]

model = Linear_ARHMM(2, 2, dyn_mtrxs, sigmas)
model.initial.density = initial_distr
model.transition.trans_mtrx = trans_mtrx

[x_track, mode_track] = model.simulate(np.random.rand(2))

mode_infer = model.viterbi(x_track)


plt.figure()
plt.subplot(211)
plt.imshow(np.array([mode_track]), aspect = 'auto')
plt.subplot(212)
plt.imshow(np.array([mode_infer]), aspect = 'auto')
plt.show()