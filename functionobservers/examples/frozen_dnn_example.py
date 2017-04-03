import os
import cPickle as pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

from functionobservers.mappers import FrozenDenseDNN


def init_regular_model(nn_layer1=100, nn_layer2=50):
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

model = init_regular_model()
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

nlayer1 = 100
nlayer2 = 50
nepochs = 1000
batch_size = 200
nn_shape = (nlayer1, nlayer2)
be_shape = (batch_size, nepochs)
print "Inferring DNN with layers " + str(nn_shape) + " over " + str(nsteps) + \
      " steps with (batch_size, nepochs) = " + str(be_shape)

# train regular model
curr_model = init_regular_model(nn_layer1=nlayer1, nn_layer2=nlayer2)
curr_data = orig_data[:, :, 0]
curr_obs = orig_obs[:, :, 0]
curr_plot_vals = orig_plot_vals[:, :, 0]
curr_model.fit(curr_data, curr_obs, batch_size=batch_size, nb_epoch=nepochs, verbose=1)
curr_preds = curr_model.predict(plot_data_in)

# train frozen model
frozen_model = FrozenDenseDNN(curr_model)
frozen_model.fit(curr_data, curr_obs)
frozen_preds = frozen_model.predict(plot_data_in)

plt.figure()
plt.plot(curr_data, curr_obs, 'ro', label='obs')
plt.plot(plot_data_in, curr_plot_vals, 'k-', linewidth=3.0, label='actual')
plt.plot(plot_data_in, curr_preds, 'b-', linewidth=3.0, label='model preds')
plt.plot(plot_data_in, frozen_preds, 'g-', linewidth=3.0, label='frozen preds')
plt.legend()

plt.show()

