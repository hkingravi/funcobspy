"""
Given a time-series, learn a global DNN model over all of the time steps: the goal here is to learn
proper representations for the first two layers. Finally, freeze the weights for the input layers,
and learn a new series of weights for the output layer per time step.
"""
import os
import cPickle as pickle
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


def init_regular_model(nn_layer1=100, nn_layer2=50):
    # construct regression model
    model = Sequential()
    model.add(Dense(nn_layer1, input_shape=(1,), init='normal', activation='relu'))
    model.add(Dense(nn_layer2, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def init_frozen_model(model_in, nn_layer1=100, nn_layer2=50):
    # construct regression model
    model = Sequential()

    # initialize layers
    layer0 = Dense(nn_layer1, input_shape=(1,), init='normal', activation='relu',
                   weights=model_in.layers[0].get_weights(), trainable=False)
    layer1 = Dense(nn_layer2, init='normal', activation='relu',
                   weights=model_in.layers[1].get_weights(), trainable=False)
    layer2 = Dense(1, init='normal')

    # add layers and compile
    model.add(layer0)
    model.add(layer1)
    model.add(layer2)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def map_forward(model_in, data):
    curr_map = model_in.layers[0].

# file and directory information
data_dir = "./data/"
out_dir = "./results/"
f_prefix = "synthetic_time_series_generator_RBFNetwork_kernel_gaussian_scheme"
f_scheme = "switching"

# create files
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
loadfile = os.path.join(data_dir, f_prefix + "_" + f_scheme + ".pkl")
savefile = os.path.join(out_dir, "synt_1d_timeseries_dnn_global_results_" + f_scheme + ".pkl")

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
print "Inferring global DNN with layers " + str(nn_shape) + " over " + str(nsteps) + \
      " steps with (batch_size, nepochs) = " + str(be_shape)

# fit global data to learn overall representation
rand_perms = np.random.permutation(nsteps*nsamp)
global_data = orig_data.reshape((data_dim, nsteps*nsamp)).T[rand_perms]
global_obs = orig_obs.reshape((obs_dim, nsteps*nsamp)).T[rand_perms]
global_model = init_regular_model(nn_layer1=nlayer1, nn_layer2=nlayer2)
global_model.fit(global_data, global_obs, batch_size=200, nb_epoch=100, verbose=1)

# now go through each step, initializing new models, and freezing the first few layers
for i in xrange(nframes):
    print "Step ", i
    curr_model = init_frozen_model(model_in=global_model, nn_layer1=nlayer1, nn_layer2=nlayer2)
    curr_model.fit(orig_data[:, :, i], orig_obs[:, :, i], batch_size=200, nb_epoch=100, verbose=1)
    pred_plot_vals[:, :, i] = curr_model.predict(plot_data_in)
    input_weight_vals[:, i] = curr_model.get_weights()[0].reshape((nlayer1,))
    mid_weight_vals[:, :, i] = curr_model.get_weights()[2]  # weights of penultimate layer
    output_weight_vals[:, i] = curr_model.get_weights()[3]  # weights of penultimate layer

print "Time taken: " + str(time.time()) + " seconds."
data_out_dict = {'plot_data': plot_data, "pred_plot_vals": pred_plot_vals,
                 'output_layer_weights': output_weight_vals,
                 'mid_layer_weights': mid_weight_vals,
                 'input_layer_weights': mid_weight_vals,}
pickle.dump(data_out_dict, open(savefile, "wb"))
