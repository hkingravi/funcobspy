"""
Given a time-series, learn a global DNN model over all of the time steps: the goal here is to learn
proper representations for the first two layers. Finally, freeze the weights for the input layers,
and learn a new series of weights for the output layer per time step.
"""
import os
import cPickle as pickle
import time
import copy
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from functionobservers.mappers import FrozenDenseDNN
from functionobservers.solvers import solve_tikhinov


def init_regular_model(input_dim, nnlayer_tuple, batchnorm_tuple, dropout_tuple, activation_tuple,
                       init_tuple, loss="mean_squared_error", optimizer="adam", use_bias=True):
    """
    See class below for description.

    :param input_dim:
    :param nnlayer_tuple:
    :param batchnorm_tuple:
    :param dropout_tuple:
    :param activation_tuple:
    :param init_tuple:
    :param loss:
    :param optimizer:
    :param use_bias:
    :return:
    """
    nlayers = set([len(nnlayer_tuple), len(batchnorm_tuple), len(dropout_tuple),
                   len(activation_tuple), len(init_tuple)])
    assert len(nlayers) == 1, "Error: all tuples must have the same length"

    # construct regression model
    model = Sequential()
    nlayers = len(nnlayer_tuple)

    for i in xrange(nlayers-1):
        if i == 0:
            model.add(Dense(nnlayer_tuple[i], input_shape=(input_dim,), kernel_initializer=init_tuple[i],
                            activation=activation_tuple[i]))
        else:
            model.add(Dense(nnlayer_tuple[i], kernel_initializer=init_tuple[i], activation=activation_tuple[i]))
        if batchnorm_tuple[i]:
            model.add(BatchNormalization())
        if dropout_tuple[i] > 0.0:
            model.add(Dropout(dropout_tuple[i]))
    last_layer = Dense(nnlayer_tuple[nlayers-1], kernel_initializer=init_tuple[nlayers-1])
    last_layer.use_bias = use_bias
    model.add(last_layer)

    # Compile model
    model.compile(loss=loss, optimizer=optimizer)
    return model


def init_frozen_model(local_model, last_layer=False):
    """
    Initialize a new frozen linearized model from the local model.

    :return:
    """
    nlayers = len(local_model.model.layers)

    # unpack model
    input_dim = local_model.input_dim
    nnlayer_tuple = local_model.nnlayer_tuple
    batchnorm_tuple = local_model.batchnorm_tuple
    dropout_tuple = local_model.dropout_tuple
    activation_tuple = local_model.activation_tuple
    init_tuple = local_model.init_tuple
    loss = local_model.loss
    optimizer = local_model.optimizer

    model = Sequential()

    for i in xrange(nlayers-1):
        curr_layer = local_model.model.layers[i]
        if i == 0:
            model.add(Dense(nnlayer_tuple[i], input_shape=(input_dim,), kernel_initializer=init_tuple[i],
                            activation=activation_tuple[i], weights=curr_layer.get_weights(), trainable=False))
        else:
            model.add(Dense(nnlayer_tuple[i], kernel_initializer=init_tuple[i],
                            activation=activation_tuple[i], weights=curr_layer.get_weights(), trainable=False))

        # not sure what happens here
        if batchnorm_tuple[i]:
            model.add(BatchNormalization())
        if dropout_tuple[i] > 0.0:
            model.add(Dropout(dropout_tuple[i]))

    if last_layer:
        # the last layer is trainable
        last_layer = Dense(nnlayer_tuple[nlayers-1], kernel_initializer=init_tuple[nlayers-1])
        last_layer.use_bias = local_model.model.layers[nlayers-1].use_bias
        model.add(last_layer)
    model.compile(loss=loss, optimizer=optimizer)
    return model


class LocalDNN(object):
    """
    Implement DNN that retrains itself for every time-step. In this, no dynamics are
    utilized, but it learns the best model locally in time. This is a regression model,
    and it's assumed that the last layer will thus be of the form Dense(1, init=normal_tuple(N)),
    where N is the number of layers. All layers utilize Keras, so it's assumed that the
    user is familiar with the library.

    """
    def __init__(self, input_dim, nnlayer_tuple, batchnorm_tuple, dropout_tuple, activation_tuple,
                 init_tuple, loss="mean_squared_error", optimizer="adam", use_bias=True):
        """

        :param input_dim: dimensionality of the input
        :param nnlayer_tuple: tuple of N integers representing number of neurons per layer
        :param batchnorm_tuple: tuple of N bools representing whether to use batch normalization
                                in that layer
        :param dropout_tuple: tuple of N floats between 0 and 1 representing amount of dropout
                              per layer
        :param activation_tuple: tuple of N strings defining activation functions per layer
        :param init_tuple: tuple of N strings representing initialization per layer
        :param loss: string representing loss
        :param optimizer: string representing optimizer
        :param use_bias: bool indicating whether to use bias in last layer
        """
        self.input_dim = input_dim
        self.nnlayer_tuple = nnlayer_tuple
        self.batchnorm_tuple = batchnorm_tuple
        self.dropout_tuple = dropout_tuple
        self.activation_tuple = activation_tuple
        self.init_tuple = init_tuple
        self.loss = loss
        self.optimizer = optimizer
        self.use_bias = use_bias
        self.model = init_regular_model(self.input_dim, self.nnlayer_tuple, self.batchnorm_tuple, self.dropout_tuple,
                                        self.activation_tuple, self.init_tuple, loss=self.loss,
                                        optimizer=self.optimizer, use_bias=self.use_bias)

    def fit(self, data, obs, **kwargs):
        """

        :param data:
        :param obs:
        :param kwargs:
        :return:
        """
        self.model.fit(data, obs, **kwargs)
        return self

    def predict(self, data, **kwargs):
        """

        :param data:
        :param kwargs:
        :return:
        """
        return self.model.predict(data, **kwargs)


class LinearizedDNN(object):
    """
    Implement DNN that retrains itself for every time-step. In this, no dynamics are
    utilized, but it learns the best model locally in time. This is a regression model,
    and it's assumed that the last layer will thus be of the form Dense(1, init=normal_tuple(N)),
    where N is the number of layers.

    Unlike the LocalDNN, however, once the overall DNN is learned, the model is reinitialized
    with the same layers frozen, and the last layer weights are the only thing that are learned,
    hence linearizing the model.

    """
    def __init__(self, local_model, optimizer="least_squares", reg_val=10.0):
        """

        :param local_model: LocalDNN instance that will be used to construct model
        """
        self.local_model = local_model
        self.optimizer = optimizer
        self.model = None
        self.weights = None
        self.reg_val = reg_val

    def transform(self, data):
        # map data manually
        mapped_data = data
        for curr_layer in self.model.layers:
            mapped_data = curr_layer.activation(np.dot(mapped_data, curr_layer.get_weights()[0])
                                                + curr_layer.get_weights()[1])
        return mapped_data

    def fit(self, data, obs, **kwargs):
        """

        :param data:
        :param obs:
        :param kwargs:
        :return:
        """
        if self.optimizer == "least_squares":
            self.model = init_frozen_model(self.local_model, last_layer=False)
            mapped_data = self.transform(data)
            self.weights = solve_tikhinov(mapped_data, obs, reg_val=self.reg_val)
        else:
            self.model = init_frozen_model(self.local_model, last_layer=True)
            self.model.fit(data, obs, **kwargs)
        return self

    def predict(self, data, **kwargs):
        """

        :param data:
        :param kwargs:
        :return:
        """
        if self.optimizer == "least_squares":
            mapped_data = self.transform(data)
            return np.dot(mapped_data, self.weights)
        else:
            return self.model.predict(data, **kwargs)


def main():
    # file and directory information
    data_dir = "./data/"
    out_dir = "./results/"
    f_prefix = "synthetic_time_series_generator_RBFNetwork_kernel_gaussian_scheme"
    f_scheme = "switching"
    debug_mode = True

    # create files
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    loadfile = os.path.join(data_dir, f_prefix + "_" + f_scheme + ".pkl")

    if debug_mode:
        savefile_local = os.path.join(out_dir, "synt_1d_timeseries_dnn_local_results_" + f_scheme + "_debug.pkl")
        savefile_linear = os.path.join(out_dir, "synt_1d_timeseries_dnn_linear_results_" + f_scheme + "_debug.pkl")
        savefile_global = os.path.join(out_dir, "synt_1d_timeseries_dnn_global_results_" + f_scheme + "_debug.pkl")
    else:
        savefile_local = os.path.join(out_dir, "synt_1d_timeseries_dnn_local_results_" + f_scheme + ".pkl")
        savefile_linear = os.path.join(out_dir, "synt_1d_timeseries_dnn_linear_results_" + f_scheme + ".pkl")
        savefile_global = os.path.join(out_dir, "synt_1d_timeseries_dnn_global_results_" + f_scheme + ".pkl")

    # load data
    data_dict = pickle.load(open(loadfile, "rb"))
    orig_obs = data_dict['orig_func_obs']
    orig_data = data_dict['orig_func_data']
    orig_plot_vals = data_dict['orig_func_plot_vals']

    # get shapes of arrays
    data_dim = orig_data.shape[0]
    obs_dim = orig_obs.shape[0]
    nsamp = orig_data.shape[1]
    nsteps = orig_data.shape[2]
    nplot = orig_plot_vals.shape[1]
    data_start = 0
    data_end = 2 * np.pi
    plot_data = np.linspace(data_start, data_end, nplot)
    plot_data_in = np.reshape(plot_data, (nplot, 1))

    # try initialization
    input_dim = 1
    nnlayer_tuple = (100, 50, 1)
    batchnorm_tuple = (False, False, False)
    dropout_tuple = (0.0, 0.0, 0.0)
    activation_tuple = ("relu", "relu", "relu")
    init_tuple = ("normal", "normal", "normal")
    loss = "mean_squared_error"
    optimizer = "adam"
    use_bias = False

    if debug_mode:
        print "Displaying regular model..."
        curr_model = LocalDNN(input_dim, nnlayer_tuple, batchnorm_tuple, dropout_tuple, activation_tuple,
                              init_tuple, loss="mean_squared_error", optimizer="adam", use_bias=True)
        for curr_layer in curr_model.model.layers:
            print type(curr_layer), "Input: ", curr_layer.input_shape, "Output: ", curr_layer.output_shape
            if isinstance(curr_layer, keras.layers.core.Dense):
                print "Init: ", curr_layer.kernel_initializer, "Units: ", curr_layer.units
            elif isinstance(curr_layer, keras.layers.core.Dropout):
                print "p: ", curr_layer.rate
            print ""
        curr_model.fit(orig_data[:, :, 0].T, orig_obs[:, :, 0].T, batch_size=200, epochs=1, verbose=0)

        print "Displaying frozen model..."
        linear_model = init_frozen_model(curr_model)
        for curr_layer in linear_model.model.layers:
            print type(curr_layer), "Input: ", curr_layer.input_shape, "Output: ", curr_layer.output_shape
            if isinstance(curr_layer, keras.layers.core.Dense):
                print "Init: ", curr_layer.kernel_initializer
            elif isinstance(curr_layer, keras.layers.core.Dropout):
                print "p: ", curr_layer.rate
            print ""
        nframes = 30
    else:
        nframes = nsteps

    # create prediction arrays
    pred_plot_vals_local = np.zeros(orig_plot_vals[:, :, 0:nframes].shape)
    pred_plot_vals_linear = np.zeros(orig_plot_vals[:, :, 0:nframes].shape)
    pred_plot_vals_global = np.zeros(orig_plot_vals[:, :, 0:nframes].shape)

    # DNN training parameters
    batch_size = 200
    nepochs = 1000
    verbose = 0

    # fit global data to learn overall representation
    print "Pretraining model for use in GlobalDNN..."
    rand_perms = np.random.permutation(nsteps * nsamp)
    global_data = orig_data.reshape((data_dim, nsteps * nsamp)).T[rand_perms]
    global_obs = orig_obs.reshape((obs_dim, nsteps * nsamp)).T[rand_perms]
    global_dnn = LocalDNN(input_dim, nnlayer_tuple, batchnorm_tuple, dropout_tuple, activation_tuple,
                          init_tuple, loss=loss, optimizer=optimizer, use_bias=use_bias)
    print global_data.shape
    global_dnn.fit(global_data, global_obs, batch_size=batch_size, nb_epoch=100, verbose=1)
    print "Finished..."

    # now go through each step, initializing new models, and freezing the first few layers
    for i in xrange(nframes):
        print "Step ", i
        print "Training LocalDNN model..."
        local_model = LocalDNN(input_dim, nnlayer_tuple, batchnorm_tuple, dropout_tuple, activation_tuple,
                               init_tuple, loss=loss, optimizer=optimizer, use_bias=use_bias)
        local_model.fit(orig_data[:, :, i].T, orig_obs[:, :, i].T, batch_size=batch_size, epochs=nepochs,
                        verbose=verbose)
        pred_plot_vals_local[:, :, i] = local_model.predict(plot_data_in).T

        print "Training LinearizedDNN model..."
        linear_model = LinearizedDNN(local_model, reg_val=100.0)
        linear_model.fit(orig_data[:, :, i].T, orig_obs[:, :, i].T)
        pred_plot_vals_linear[:, :, i] = linear_model.predict(plot_data_in).T

        print "Training LinearizedDNN model..."
        global_model = LinearizedDNN(global_dnn, reg_val=100.0)
        global_model.fit(orig_data[:, :, i].T, orig_obs[:, :, i].T)
        pred_plot_vals_global[:, :, i] = global_model.predict(plot_data_in).T

    # dump results
    data_out_dict_local = {'plot_data': plot_data, "pred_plot_vals": pred_plot_vals_local,
                           'all_weights': 0}
    data_out_dict_linear = {'plot_data': plot_data, "pred_plot_vals": pred_plot_vals_linear,
                            'all_weights': 0}
    data_out_dict_global = {'plot_data': plot_data, "pred_plot_vals": pred_plot_vals_global,
                            'all_weights': 0}
    pickle.dump(data_out_dict_local, open(savefile_local, "wb"))
    pickle.dump(data_out_dict_linear, open(savefile_linear, "wb"))
    pickle.dump(data_out_dict_global, open(savefile_global, "wb"))


if __name__ == '__main__':
    main()

