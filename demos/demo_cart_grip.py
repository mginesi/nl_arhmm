import numpy as np
from nl_arhmm.arhmm import Hand_Gripper_ARHMM
import matplotlib.pyplot as plt

model = Hand_Gripper_ARHMM(3, 1)

model.dynamics[0].weights_hand[0] = np.array([
    [1 ,0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
    ])*0.01
model.dynamics[0].weights_gripper[0] = np.array([[0, 0, 1]]) * 0.1
model.dynamics[0].covariance_hand[0] = 0.01 * np.eye(3)
model.dynamics[0].covariance_gripper[0] = 0.01

model.dynamics[1].weights_hand[0] = np.array([
    [1 ,0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0]
    ])*0.1
model.dynamics[1].weights_gripper[0] = np.array([[1, 0, 0]]) * 0.1
model.dynamics[1].covariance_hand[0] = 0.01 * np.eye(3)
model.dynamics[1].covariance_gripper[0] = 0.01

model.dynamics[2].weights_hand[0] = np.array([
    [0 ,0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1]
    ])*0.1
model.dynamics[2].weights_gripper[0] = np.array([[1, 0, 0]]) * 0.1
model.dynamics[2].covariance_hand[0] = 0.01 * np.eye(3)
model.dynamics[2].covariance_gripper[0] = 0.01

[x_track, lbl_track] = model.simulate()

fig = plt.figure()
ax = plt.axes(projection='3d')
plt.plot(x_track[:,0], x_track[:,1], x_track[:,2])

fig2 = plt.figure()
plt.plot(x_track[:,3])

plt.show()
