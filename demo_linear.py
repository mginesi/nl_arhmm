import numpy as np
from arhmm import Linear_ARHMM
import matplotlib.pyplot as plt

# ARHMM

# True model
dyn_mtrxs = [0.5 * np.random.rand(2,3), 0.5 * np.random.rand(2,3)]
sigmas = []
for _n in range(4):
    rm = np.eye(2)
    sigmas.append(rm)
model_true = Linear_ARHMM(2, 2, dyn_mtrxs, sigmas)

[x_true, mode_true] = model_true.simulate(np.random.rand(2))
mode_infered_true = model_true.viterbi(x_true)

# Test model
dyn_mtrxs = [0.5 * np.random.rand(2,3), 0.5 * np.random.rand(2,3)]
sigmas = []
for _n in range(4):
    rm = np.eye(2)
    sigmas.append(rm)
model = Linear_ARHMM(2, 2, dyn_mtrxs, sigmas)
model.em_algorithm(x_true)
mode_infered = model.viterbi(x_true)

plt.figure()
plt.subplot(311)
plt.imshow(np.array([mode_true]), aspect='auto')
plt.subplot(312)
plt.imshow(np.array([mode_infered_true]), aspect='auto')
plt.subplot(313)
plt.imshow(np.array([mode_infered]), aspect='auto')
plt.show()