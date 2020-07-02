import numpy as np
from nl_arhmm import Linear_ARHMM
import matplotlib.pyplot as plt

# True trajectory is a square

interval = np.linspace(0, 1, 50)
bottom = np.array([interval, np.zeros(50)]).transpose()
right = np.array([np.ones(50), interval]).transpose()
top = np.array([np.flip(interval), np.ones(50)]).transpose()
left = np.array([np.zeros(50), np.flip(interval)]).transpose()

data = np.concatenate([bottom, right, top, left])

# ARHMM
dyn_mtrxs = [10 * np.random.rand(2,3), 10 * np.random.rand(2,3), 10 * np.random.rand(2,3), 10 * np.random.rand(2,3)]
sigmas = []
for _n in range(4):
    rm = np.random.rand(2, 2)
    sigmas.append(np.dot(rm, rm.transpose()))
model = Linear_ARHMM(2, 4, dyn_mtrxs, sigmas)

model.em_algorithm(data)

mode_inferred = model.viterbi(data)

plt.figure()
plt.imshow(np.array([mode_inferred]), aspect='auto')
plt.show()