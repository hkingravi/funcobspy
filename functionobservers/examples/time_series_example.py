import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functionobservers.mappers import RBFNetwork
from functionobservers.utils import time_varying_uncertainty, pack_state, unpack_state


def animate_orig_data(ii):
    orig_line.set_xdata(pack_state(data))
    orig_line.set_ydata(pack_state(orig_func_obs[:, :, ii]))
    plot_line.set_xdata(pack_state(eval_data))
    plot_line.set_ydata(pack_state(orig_func_plot_vals[:, :, ii]))
    return orig_line, plot_line


# set seed
np.random.seed(0)

# time-varying data arises from weight set associated to an RBF network
orig_function_dim = 5
init_weights_orig = np.random.randn(orig_function_dim, 1)
weights = init_weights_orig
dt = 0.03
current_t = 0
final_time = 20
nsteps = int(np.floor(final_time / dt))
data_start = 0
data_end = 2*np.pi

eval_data = np.linspace(data_start, data_end, nsteps).reshape((1, nsteps))  # plotting
centers_orig = np.linspace(data_start, data_end, orig_function_dim).reshape((1, orig_function_dim))
params = np.array([0.3])
noise = 0.1
model = RBFNetwork(centers=centers_orig, k_func="gaussian", params=params, noise=noise)

# data
nsamp = 500
nsamp_plot = nsteps  # number of samples for plotting
data = np.linspace(data_start, data_end, nsamp).reshape((1, nsamp))  # where the function values are sampled from

# run simulation
scheme = "switching"
times = np.zeros((nsteps,))
ideal_weight_trajectory = np.zeros((orig_function_dim, nsteps))
orig_func_vals = np.zeros((1, nsamp, nsteps))
orig_func_plot_vals = np.zeros((1, nsamp_plot, nsteps))
orig_func_data = np.zeros((1, nsamp, nsteps))
orig_func_obs = np.zeros((1, nsamp, nsteps))

# compute times; used to generate time-varying function
for i in xrange(nsteps):
    times[i] = current_t
    current_t = dt*i

for i in xrange(nsteps):
    weights = time_varying_uncertainty(weights_star=weights, t=times[i], scheme=scheme)

    orig_func = model.predict(data, weights_in=weights)
    orig_func_plot = model.predict(eval_data, weights_in=weights)
    orig_obs = orig_func + noise*np.random.randn(orig_func.shape[0], orig_func.shape[1])

    # store function values for plotting
    ideal_weight_trajectory[:, i] = pack_state(weights)
    orig_func_vals[:, :, i] = pack_state(orig_func)
    orig_func_plot_vals[:, :, i] = pack_state(orig_func_plot)
    orig_func_data[:, :, i] = data
    orig_func_obs[:, :, i] = orig_obs

# dump data
data_dir = ""
f_prefix = "synthetic_time_series_generator_RBFNetwork_kernel_gaussian_scheme"
savefile = os.path.join(data_dir, f_prefix + "_" + scheme + ".pkl")
data_dict = {'orig_func_vals': orig_func_vals, 'orig_func_plot_vals': orig_func_plot_vals,
             'orig_func_data': orig_func_data, 'orig_func_obs': orig_func_obs}
pickle.dump(data_dict, open(savefile, "wb"))

# plot data
plt.plot(ideal_weight_trajectory.T)
plt.title("Weights for " + scheme + " scheme")

# plot function
nframes = nsteps
fig = plt.figure()
fig.suptitle('Original function and observations', fontsize=15)
ax = fig.add_subplot(111)
orig_line, = ax.plot([], [], 'ro', label='observations', markersize=10)
plot_line, = ax.plot([], [], 'b-', label='function', linewidth=3.0)
ax.set_xlim(data_start, data_end)
ax.set_ylim(-3, 3)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
ani_orig = animation.FuncAnimation(fig, animate_orig_data, frames=nframes, fargs=(),
                                   interval=100, blit=True)

plt.show()
