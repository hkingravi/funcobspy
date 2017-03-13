import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.io import loadmat


def animate_orig_data(i):
    orig_line.set_xdata(np.squeeze(orig_data[:, :, i]))
    orig_line.set_ydata(np.squeeze(orig_obs[:, :, i]))
    plot_line.set_xdata(plot_data)
    plot_line.set_ydata(np.squeeze(orig_plot_vals[:, :, i]))
    return orig_line, plot_line

m_filename = "./data/synthetic_time_series_generator_RBFNetwork_kernel_gaussian_scheme_switching.mat"
p_filename = "./data/synthetic_time_series_generator_RBFNetwork_kernel_gaussian_scheme_switching.pkl"
data_dict = loadmat(m_filename)
orig_obs = data_dict['orig_func_obs']
orig_data = data_dict['orig_func_data']
orig_plot_vals = data_dict['orig_func_plot_vals']

# get shapes of arrays to construct new shapes
data_dim = orig_data.shape[0]
obs_dim = orig_obs.shape[0]
plot_dim = orig_plot_vals.shape[0]
nsamp = orig_data.shape[1]
nsteps = orig_data.shape[2]
nplot = orig_plot_vals.shape[1]

# construct new arrays to recast in terms of row major, and save the file
new_obs = np.zeros((nsamp, obs_dim, nsteps))
new_data = np.zeros((nsamp, data_dim, nsteps))
new_plot_vals = np.zeros((nplot, plot_dim, nsteps))
data_out_dict = {'orig_func_obs': new_obs, 'orig_func_data': new_data, 'orig_func_plot_vals': new_plot_vals}

for i in xrange(nsteps):
    data_out_dict['orig_func_obs'][:, :, i] = np.reshape(data_dict['orig_func_obs'][:, :, i], (nsamp, obs_dim))
    data_out_dict['orig_func_data'][:, :, i] = np.reshape(data_dict['orig_func_data'][:, :, i], (nsamp, data_dim))
    data_out_dict['orig_func_plot_vals'][:, :, i] = \
        np.reshape(data_dict['orig_func_plot_vals'][:, :, i], (nplot, plot_dim))

pickle.dump(data_out_dict, open(p_filename, "wb"))

# plot outputs
data_start = 0
data_end = 2*np.pi
plot_data = np.linspace(data_start, data_end, nplot)

print "Training data: " + str(nsamp) + " " + str(orig_data.shape[0]) + \
      "-dimensional samples over " + str(nsteps) + " time steps."
print "Training observations: " + str(nsamp) + " " + str(orig_obs.shape[0]) + \
      "-dimensional samples over " + str(nsteps) + " time steps."
print "Training plot observations: " + str(nplot) + " " + str(orig_plot_vals.shape[0]) + \
      "-dimensional samples over " + str(nsteps) + " time steps."

fig = plt.figure()
ax = fig.add_subplot(111)
orig_line, = ax.plot([], [], 'ro', label='observations', markersize=10)
plot_line, = ax.plot([], [], 'b-', label='function')
ax.set_xlim(data_start, data_end)
ax.set_ylim(-5, 5)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
fig.suptitle('test title', fontsize=20)

ani = animation.FuncAnimation(fig, animate_orig_data, frames=nsteps, fargs=(),
                              interval=100, blit=True)
plt.show()
