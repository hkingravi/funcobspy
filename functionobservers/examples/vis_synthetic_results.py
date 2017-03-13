import os
import cPickle as pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from keras.models import Sequential
from keras.layers import Dense


def animate_orig_data(ii):
    orig_line.set_xdata(np.squeeze(orig_data[:, :, ii]))
    orig_line.set_ydata(np.squeeze(orig_obs[:, :, ii]))
    plot_line.set_xdata(plot_data)
    plot_line.set_ydata(np.squeeze(orig_plot_vals[:, :, ii]))
    return orig_line, plot_line


def animate_pred_data(ii):
    plot_line2.set_xdata(plot_data)
    plot_line2.set_ydata(np.squeeze(orig_plot_vals[:, :, ii]))
    pred_line.set_xdata(plot_data)
    pred_line.set_ydata(np.squeeze(pred_plot_vals[:, :, ii]))
    return plot_line2, pred_line

# file and directory information
data_dir = "./data/"
results_dir = "./results/"
f_prefix = "synthetic_time_series_generator_RBFNetwork_kernel_gaussian_scheme"
f_scheme = "switching"

# load files
loadfile_orig = os.path.join(data_dir, f_prefix + "_" + f_scheme + ".pkl")
loadfile = os.path.join(results_dir, "synt_1d_timeseries_dnn_results_" + f_scheme + ".pkl")

orig_data_dict = pickle.load(open(loadfile_orig, "rb"))
data_dict = pickle.load(open(loadfile, "rb"))

orig_obs = orig_data_dict['orig_func_obs']
orig_data = orig_data_dict['orig_func_data']
orig_plot_vals = orig_data_dict['orig_func_plot_vals']

# get shapes of arrays
data_dim = orig_data.shape[1]
obs_dim = orig_obs.shape[1]
plot_dim = orig_plot_vals.shape[1]
nsamp = orig_data.shape[0]
nsteps = orig_data.shape[2]
nplot = orig_plot_vals.shape[0]
data_start = 0
data_end = 2*np.pi
plot_data = np.linspace(data_start, data_end, nplot)
plot_data_in = np.reshape(plot_data, (nplot, 1))
pred_plot_vals = np.zeros(orig_plot_vals.shape)
plot_data = data_dict['plot_data']
pred_plot_vals = data_dict['pred_plot_vals']
output_weight_vals = data_dict['output_layer_weights']
mid_weight_vals = data_dict['mid_layer_weights']

# plotting
nframes = nsteps
show_orig = True
show_pred = True

print mid_weight_vals.shape
mid_weight_vals = mid_weight_vals.reshape(5000, 666)

if show_orig:
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

if show_pred:
    fig2 = plt.figure()
    fig2.suptitle('Original function and predictions', fontsize=15)
    ax2 = fig2.add_subplot(111)
    plot_line2, = ax2.plot([], [], 'b', label='function', linewidth=3.0)
    pred_line, = ax2.plot([], [], 'g-', label='predicted', linewidth=3.0)
    ax2.set_xlim(data_start, data_end)
    ax2.set_ylim(-3, 3)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels)
    ani_pred = animation.FuncAnimation(fig2, animate_pred_data, frames=nframes, fargs=(),
                                       interval=100, blit=True)

fig3 = plt.figure()
for i in xrange(output_weight_vals.shape[0]):
    plt.plot(output_weight_vals[i, :])
fig3.suptitle('Final layer weight evolution', fontsize=15)

fig4 = plt.figure()
for i in xrange(100):
    plt.plot(mid_weight_vals[i, :])
fig4.suptitle('Mid layer weight evolution', fontsize=15)

plt.show()
