import os
import cPickle as pickle
import time
import theano as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from keras.models import Sequential
from keras.layers import Dense


def map_data(model_in, data):
    """
    Given a DNN keras model with dense layers all the way through, map the data until
    the penultimate layer, before the final summation.

    :param model_in: keras DNN model with Dense layers.
    :param data: data of shape (N, D).
    :return:
    """
    mapped_data = data
    nlayers = len(model_in.layers)
    for i in xrange(nlayers-1):
        curr_layer = model_in.layers[i]
        mapped_data = curr_layer.activation(np.dot(mapped_data, curr_layer.get_weights()[0])
                                            + curr_layer.get_weights()[1])
    return mapped_data


def map_full(model_in, data):
    """
    Given a DNN keras model with dense layers all the way through, map the data until
    the penultimate layer, before the final summation.

    :param model_in: keras DNN model with Dense layers.
    :param data: data of shape (N, D).
    :return:
    """
    mapped_data = data
    nlayers = len(model_in.layers)
    for i in xrange(nlayers):
        curr_layer = model_in.layers[i]
        print "Current layer: ", i, ": type =" + str(curr_layer)
        mapped_data = curr_layer.activation(np.dot(mapped_data, curr_layer.get_weights()[0])
                                            + curr_layer.get_weights()[1])
    return mapped_data

def init_model(nn_layer1=100, nn_layer2=50):
    # construct regression model
    model_out = Sequential()
    model_out.add(Dense(nn_layer1, input_shape=(1,), init='normal', activation='relu'))
    model_out.add(Dense(nn_layer2, init='normal', activation='relu'))
    model_out.add(Dense(1, init='normal'))

    # Compile model
    model_out.compile(loss='mean_squared_error', optimizer='adam')
    return model_out


np.random.seed(0)
nlayer1 = 100
nlayer2 = 50
model = Sequential()
model.add(Dense(nlayer1, input_shape=(1,), init='normal', activation='relu'))
model.add(Dense(nlayer2, init='normal', activation='relu'))
model.add(Dense(1, init='normal'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# file and directory information
data_dir = "./data/"
out_dir = "./results/"
f_prefix = "synthetic_time_series_generator_RBFNetwork_kernel_gaussian_scheme"
f_scheme = "switching"

# create files
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
loadfile = os.path.join(data_dir, f_prefix + "_" + f_scheme + ".pkl")

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

input_weight_vals = np.zeros((nlayer1, nsteps))
mid_weight_vals = np.zeros((nlayer1, nlayer2, nsteps))
output_weight_vals = np.zeros((nlayer2, nsteps))

nframes = nsteps
for i in xrange(1):
    print "Step ", i
    curr_model = init_model(nn_layer1=nlayer1, nn_layer2=nlayer2)
    curr_model.fit(orig_data[:, :, i], np.squeeze(orig_obs[:, :, i]), batch_size=200, nb_epoch=1000, verbose=0)
    pred_plot_vals[:, :, i] = curr_model.predict(plot_data_in)
    input_weight_vals[:, i] = curr_model.get_weights()[0].reshape((nlayer1,))
    mid_weight_vals[:, :, i] = curr_model.get_weights()[2]  # weights of penultimate layer
    output_weight_vals[:, i] = curr_model.get_weights()[3]  # weights of penultimate layer


curr_data = orig_data[:, :, 0]
curr_obs = np.squeeze(orig_obs[:, :, 0])

# construct map function
get_activations = T.function([curr_model.layers[0].input], curr_model.layers[1].output)
final_map = T.function([curr_model.layers[2].input], curr_model.layers[2].output)

sort_inds = np.argsort(np.squeeze(curr_data))
sorted_data = curr_data[sort_inds, :]
sorted_preds = curr_model.predict(sorted_data)

mapped_data1 = map_data(curr_model, sorted_data.astype(np.float32))
mapped_data2 = get_activations(sorted_data.astype(np.float32))
mapped_preds1 = final_map(mapped_data1.astype(np.float32))
mapped_preds2 = np.dot(mapped_data1, curr_model.layers[2].get_weights()[0])# + curr_model.layers[2].get_weights()[1]
mapped_preds3 = final_map(mapped_data2.astype(np.float32))
mapped_preds4 = map_full(curr_model, sorted_data.astype(np.float32))
#sorted_m_preds2 = np.sort(mapped_preds2)

plt.figure()
plt.plot(np.squeeze(sorted_data), sorted_preds, 'b-', label="original")
plt.plot(np.squeeze(sorted_data), mapped_preds1, 'g-', label="preds1")
plt.plot(np.squeeze(sorted_data), mapped_preds2, 'r-', label="preds2")
plt.plot(np.squeeze(sorted_data), mapped_preds4, 'k-', label="preds4")
plt.title("Predictions on sorted data using different maps")
plt.legend()

plt.figure()
plt.imshow(mapped_data2)
plt.title("Feature mapped data")

plt.show()

