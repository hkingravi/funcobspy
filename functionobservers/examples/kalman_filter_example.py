"""
Plot an example of a Kalman filter.
"""

import numpy as np
import matplotlib.pyplot as plt
from functionobservers.filters import KalmanFilter
from functionobservers.utils import pack_state, unpack_state

# plot parameters
f_lwidth = 3
f_marksize = 2.0
c_marksize = 15
font_size = 15
save_results = True
ext = 'png'

# set seed
np.random.seed(20)
add_process_noise = True

# filter parameters
Ac = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])  # dynamics model for radar
C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

dyn_dim = Ac.shape[0]
meas_dim = C.shape[0]

dt = 0.001  # time step
A = np.eye(dyn_dim) + dt*Ac  # discretize dynamics operator

nmeas = C.shape[0]
ncent = A.shape[0]

P_init = 0.0001*np.eye(ncent)
Q = 0.0001*np.eye(ncent)
R = 0.1*np.eye(nmeas)
QL = np.linalg.cholesky(Q)
RL = np.linalg.cholesky(R)
m_init = np.array([[2, 0.001, -3, -0.01]]).T
rand_init = np.random.randn(dyn_dim, 1)

# initialize KalmanFilter, and compute filter measurements and corrections
kf = KalmanFilter(P_init, Q, R)
kf.fit(A, C, rand_init)

time_steps = 1000
states_noisy = np.zeros((ncent, time_steps))
meas_noisy = np.zeros((nmeas, time_steps))
meas_actual = np.zeros((nmeas, time_steps))
meas_kalman = np.zeros((nmeas, time_steps))
curr_state = m_init

# generate measurements
for i in xrange(time_steps):
    meas_noisy[:, i] = pack_state(np.dot(C, curr_state) + np.dot(RL, np.random.randn(nmeas, 1)))
    meas_actual[:, i] = pack_state(np.dot(C, curr_state))

    pred_state, pred_P = kf.predict(unpack_state(meas_noisy[:, i]))
    meas_kalman[:, i] = pack_state(np.dot(C, pred_state))

    if add_process_noise:
        states_noisy[:, i] = pack_state(curr_state + np.dot(QL, np.random.randn(ncent, 1)))
    else:
        states_noisy[:, i] = pack_state(curr_state)

    curr_state = np.dot(A, curr_state)

# plot final results
fig = plt.figure()
ax = fig.add_subplot(111)
fig.suptitle('Measurements and recovery of x position of aircraft from radar measurements', fontsize=10)
ax.plot(meas_noisy[0, :], 'ro', label="measured", linewidth=1.0, markerfacecolor='white')
ax.plot(meas_actual[0, :], 'b-', label="actual", linewidth=f_lwidth)
ax.plot(meas_kalman[0, :], 'g-', label="kalman", linewidth=f_lwidth)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
ax.set_xlabel('Time steps')
ax.set_ylabel('x position')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
fig2.suptitle('Measurements and recovery of y position of aircraft from radar measurements', fontsize=10)
ax2.plot(meas_noisy[1, :], 'ro', label="measured", linewidth=1.0, markerfacecolor='white')
ax2.plot(meas_actual[1, :], 'b-', label="actual", linewidth=f_lwidth)
ax2.plot(meas_kalman[1, :], 'g-', label="kalman", linewidth=f_lwidth)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels)
ax2.set_xlabel('Time steps')
ax2.set_ylabel('x position')

plt.show()
