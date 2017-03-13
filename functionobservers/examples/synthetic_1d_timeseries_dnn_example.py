"""

"""
import os
import cPickle as pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from keras.models import Sequential
from keras.layers import Dense


def init_model(nn_layer1=100, nn_layer2=50):
    # construct regression model
    model = Sequential()
    model.add(Dense(nn_layer1, input_shape=(1,), init='normal', activation='relu'))
    model.add(Dense(nn_layer2, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


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
out_dir = "./results/"
f_prefix = "synthetic_time_series_generator_RBFNetwork_kernel_gaussian_scheme"
f_scheme = "switching"

# create files
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
loadfile = os.path.join(data_dir, f_prefix + "_" + f_scheme + ".pkl")
savefile = os.path.join(out_dir, "synt_1d_timeseries_dnn_results_" + f_scheme + ".pkl")

data_dict = pickle.load(open(loadfile, "rb"))
orig_obs = data_dict['orig_func_obs']
orig_data = data_dict['orig_func_data']
orig_plot_vals = data_dict['orig_func_plot_vals']

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

nlayer1 = 100
nlayer2 = 50
mid_weight_vals = np.zeros((nlayer1, nlayer2, nsteps))
output_weight_vals = np.zeros((nlayer2, nsteps))

nframes = nsteps
for i in xrange(nframes):
    print "Step ", i
    curr_model = init_model(nn_layer1=nlayer1, nn_layer2=nlayer2)
    curr_model.fit(orig_data[:, :, i], orig_obs[:, :, i], batch_size=200, nb_epoch=1000, verbose=0)
    pred_plot_vals[:, :, i] = curr_model.predict(plot_data_in)
    mid_weight_vals[:, :, i] = curr_model.get_weights()[2]  # weights of penultimate layer
    output_weight_vals[:, i] = curr_model.get_weights()[3]  # weights of penultimate layer

# plotting
fig = plt.figure()
ax = fig.add_subplot(111)
orig_line, = ax.plot([], [], 'ro', label='observations', markersize=10)
plot_line, = ax.plot([], [], 'b-', label='function', linewidth=3.0)
ax.set_xlim(data_start, data_end)
ax.set_ylim(-5, 5)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

fig2 = plt.figure()
fig2.suptitle('Original Function and Predictions', fontsize=20)
ax2 = fig2.add_subplot(111)
plot_line2, = ax2.plot([], [], 'b', label='function', linewidth=3.0)
pred_line, = ax2.plot([], [], 'g-', label='predicted', linewidth=3.0)
ax2.set_xlim(data_start, data_end)
ax2.set_ylim(-3, 3)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels)

fig3 = plt.figure()
for i in xrange(nlayer2):
    plt.plot(output_weight_vals[i, :])

ani_orig = animation.FuncAnimation(fig, animate_orig_data, frames=nframes, fargs=(),
                                   interval=10, blit=True)
ani_pred = animation.FuncAnimation(fig2, animate_pred_data, frames=nframes, fargs=(),
                                   interval=10, blit=True)
data_out_dict = {'plot_data': plot_data, "pred_plot_vals": pred_plot_vals,
                 'output_layer_weights': output_weight_vals, 'mid_layer_weights': mid_weight_vals}
pickle.dump(data_out_dict, open(savefile, "wb"))

plt.show()
