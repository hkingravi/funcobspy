"""
Given a time-series, learn a DNN model every time step, and output the results.
"""
import os
import cPickle as pickle
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from functionobservers.utils import map_data


def init_model(nn_layer1=100, nn_layer2=50):
    # construct regression model
    model = Sequential()
    model.add(Dense(nn_layer1, input_shape=(1,), init='normal', activation='relu'))
    model.add(Dense(nn_layer2, init='normal', activation='relu'))
    last_layer = Dense(1, init='normal')
    last_layer.use_bias = False
    model.add(last_layer)

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

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
input_weight_vals = np.zeros((nlayer1, nsteps))
mid_weight_vals = np.zeros((nlayer1, nlayer2, nsteps))
output_weight_vals = np.zeros((nlayer2, nsteps))

nframes = nsteps
s_t = time.time()

nepochs = 1000
batch_size = 200
nn_shape = (nlayer1, nlayer2)
be_shape = (batch_size, nepochs)
all_weights = []
print "Inferring DNN with layers " + str(nn_shape) + " over " + str(nsteps) + \
      " steps with (batch_size, nepochs) = " + str(be_shape)
for i in xrange(nframes):
    print "Step ", i
    curr_model = init_model(nn_layer1=nlayer1, nn_layer2=nlayer2)
    curr_model.fit(orig_data[:, :, i], orig_obs[:, :, i], batch_size=200, nb_epoch=1000, verbose=0)
    pred_plot_vals[:, :, i] = curr_model.predict(plot_data_in)
    input_weight_vals[:, i] = curr_model.get_weights()[0].reshape((nlayer1,))
    mid_weight_vals[:, :, i] = curr_model.get_weights()[2]  # weights of penultimate layer
    output_weight_vals[:, i] = curr_model.get_weights()[3]  # weights of penultimate layer
    all_weights.append(curr_model.get_weights())

print "Time taken: " + str(time.time()) + " seconds."
data_out_dict = {'plot_data': plot_data, "pred_plot_vals": pred_plot_vals,
                 'all_weights': all_weights}
pickle.dump(data_out_dict, open(savefile, "wb"))
