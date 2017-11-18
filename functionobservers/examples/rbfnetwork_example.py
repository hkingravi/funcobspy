"""

"""
import numpy as np
import matplotlib.pyplot as plt

# make data
np.random.seed(0)
nsamp = 1000
noise = 0.1
data_in_1D = np.reshape(np.linspace(-2, 2, num=nsamp), (nsamp, 1))
func = np.sin(data_in_1D) + 0.1*np.power(data_in_1D, 2)
func_obs = func + noise*np.random.randn(nsamp, 1)

# construct training and validation sets
rand_inds = np.random.permutation(nsamp)
data_v = data_in_1D[rand_inds, :]
func_obs_v = func_obs[rand_inds, :]
data_in = {'X_train': data_v[0:900, :],
           'y_train': func_obs_v[0:900, :],
           'X_val': data_v[901:100, :],
           'y_val': func_obs_v[901:100, :]}


fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax1.plot(np.squeeze(data_in_1D), np.squeeze(func_obs), 'ro', linewidth=3.0, label="obs")
ax1.set_xlabel("x")
ax1.set_ylabel("f(x)")
ax1.title.set_text("Function with observations")
plt.show()
