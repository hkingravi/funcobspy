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


def animate_local_pred_data(ii):
    plot_line2.set_xdata(plot_data)
    plot_line2.set_ydata(np.squeeze(orig_plot_vals[:, :, ii]))
    pred_line.set_xdata(plot_data)
    pred_line.set_ydata(np.squeeze(pred_plot_vals[:, :, ii]))
    return plot_line2, pred_line


def animate_linear_pred_data(ii):
    plot_line_lin.set_xdata(plot_data)
    plot_line_lin.set_ydata(np.squeeze(orig_plot_vals[:, :, ii]))
    pred_line_lin.set_xdata(plot_data)
    pred_line_lin.set_ydata(np.squeeze(pred_linear_plot_vals[:, :, ii]))
    return plot_line_lin, pred_line_lin


def animate_global_pred_data(ii):
    plot_line3.set_xdata(plot_data)
    plot_line3.set_ydata(np.squeeze(orig_plot_vals[:, :, ii]))
    pred_line2.set_xdata(plot_data)
    pred_line2.set_ydata(np.squeeze(pred_global_plot_vals[:, :, ii]))
    return plot_line3, pred_line2

# file and directory information
data_dir = "./data/"
results_dir = "./results/"
f_prefix = "synthetic_time_series_generator_RBFNetwork_kernel_gaussian_scheme"
f_scheme = "switching"

show_orig = True
show_local_pred = True
show_global_pred = True
show_linear_pred = True
show_final_weights = False
show_mid_layer_weights = False


# load files
loadfile_orig = os.path.join(data_dir, f_prefix + "_" + f_scheme + ".pkl")
loadfile_local = os.path.join(results_dir, "synt_1d_timeseries_dnn_results_" + f_scheme + ".pkl")

orig_data_dict = pickle.load(open(loadfile_orig, "rb"))
data_dict = pickle.load(open(loadfile_local, "rb"))

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
output_local_weight_vals = data_dict['output_layer_weights']
mid_weight_vals = data_dict['mid_layer_weights']

if show_linear_pred:
    loadfile_linear = os.path.join(results_dir, "synt_1d_timeseries_dnn_linear_results_" + f_scheme + ".pkl")
    data_linear_dict = pickle.load(open(loadfile_linear, "rb"))
    pred_linear_plot_vals = data_linear_dict['pred_plot_vals']
    output_linear_weight_vals = data_linear_dict['frozen_weights']

if show_global_pred:
    loadfile_global = os.path.join(results_dir, "synt_1d_timeseries_dnn_global_results_" + f_scheme + ".pkl")
    data_global_dict = pickle.load(open(loadfile_global, "rb"))
    pred_global_plot_vals = data_global_dict['pred_plot_vals']
    output_global_weight_vals = data_global_dict['frozen_weights']


# plotting
nframes = nsteps
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

if show_local_pred:
    fig2 = plt.figure()
    fig2.suptitle('Original function and predictions (local vs global)', fontsize=15)
    ax2 = fig2.add_subplot(131)
    plot_line2, = ax2.plot([], [], 'b', label='function', linewidth=3.0)
    pred_line, = ax2.plot([], [], 'g-', label='local', linewidth=3.0)
    ax2.set_xlim(data_start, data_end)
    ax2.set_ylim(-3, 3)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels)
    ani_pred = animation.FuncAnimation(fig2, animate_local_pred_data, frames=nframes, fargs=(),
                                       interval=100, blit=True)

if show_linear_pred:
    ax_lin = fig2.add_subplot(132)
    plot_line_lin, = ax_lin.plot([], [], 'b', label='function', linewidth=3.0)
    pred_line_lin, = ax_lin.plot([], [], 'g-', label='linear', linewidth=3.0)
    ax_lin.set_xlim(data_start, data_end)
    ax_lin.set_ylim(-3, 3)
    handles, labels = ax_lin.get_legend_handles_labels()
    ax_lin.legend(handles, labels)
    ani_pred_lin = animation.FuncAnimation(fig2, animate_linear_pred_data, frames=nframes, fargs=(),
                                           interval=100, blit=True)


if show_global_pred:
    #fig3 = plt.figure()
    #fig3.suptitle('Original function and predictions (global)', fontsize=15)
    ax3 = fig2.add_subplot(133)
    plot_line3, = ax3.plot([], [], 'b', label='function', linewidth=3.0)
    pred_line2, = ax3.plot([], [], 'g-', label='global', linewidth=3.0)
    ax3.set_xlim(data_start, data_end)
    ax3.set_ylim(-3, 3)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels)
    ani_pred2 = animation.FuncAnimation(fig2, animate_global_pred_data, frames=nframes, fargs=(),
                                        interval=100, blit=True)

if show_final_weights:
    fig4 = plt.figure()
    ax_w1 = fig4.add_subplot(121)
    ax_w2 = fig4.add_subplot(122)
    for i in xrange(output_local_weight_vals.shape[0]):
        ax_w1.plot(output_local_weight_vals[i, :])
        ax_w2.plot(output_global_weight_vals[i, :])

    fig4.suptitle('Final layer weight evolution (local vs global)', fontsize=15)

if show_mid_layer_weights:
    fig5 = plt.figure()
    for i in xrange(100):
        plt.plot(mid_weight_vals[i, :])
    fig5.suptitle('Mid layer weight evolution', fontsize=15)

plt.show()
