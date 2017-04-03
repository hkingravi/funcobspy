"""
Classes for mapping data from an input domain to a different feature space.
Note that mappers must be trained.
"""
from abc import ABCMeta, abstractmethod
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from functionobservers.custom_exceptions import *
from functionobservers.mappers import KernelType, kernel
from functionobservers.solvers import solve_tikhinov


class Mapper:
    """
    .. codeauthor:: Hassan A. Kingravi <hkingravi@gmail.com>

    This class needs to be able to map input data to a transformed feature space
    where the dynamics model can be learned. Even though it's something of a misnomer,
    we require the class to be able to compute predictions as well.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, data, obs, **kwargs):
        pass

    @abstractmethod
    def transform(self, data, **kwargs):
        pass

    @abstractmethod
    def predict(self, data, **kwargs):
        pass


class DNN(Mapper):
    def __init__(self, **kwargs):
        """
        Implement a deep neural network-based mapper. It's assumed that the user knows the Keras
        API and can add layers using the add function, and add an optimizer for the DNN using
        the compile function.
        """
        self.model = Sequential(kwargs)

    def add(self, layer):
        """
        Add a layer in the model.

        :param layer:
        :return:
        """
        self.model.add(layer)

    def compile(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        self.model.compile(**kwargs)

    def transform(self, data, **kwargs):
        pass

    def fit(self, data, obs, **kwargs):
        self.model.fit(data, obs, **kwargs)
        return self

    def predict(self, data, **kwargs):
        return self.model.predict(data, **kwargs)

    def score(self, data, labels, **kwargs):
        return self.model.evaluate(data, labels, **kwargs)


class FrozenDenseDNN(Mapper):
    def __init__(self, model_in):
        """
        Implement a deep neural network-based mapper, based on a previously trained DNN. The structure
        of the DNN is assumed to be a series of Dense keras layers terminating in a final Dense layer
        with a single-dimensional linear output for regression (i.e. it's been trained with mean-squared
        error as the objective function) WITH NO BIAS. This class will read off all of the weights of this
        model except the final layer's, set each layer's type to be the same as the original, set the weights,
        and freeze them. Then, when new data comes in, it will map the data to a DNN space of representation,
        and perform least squares to fit the model.

        :param model_in: DNN model in the form of a keras Sequential object with multiple Dense layers.
        """
        nlayers = len(model_in.layers)
        model = Sequential()
        input_layer = Dense(model_in.layers[0].output_dim, input_shape=(1,), init='normal', activation='relu',
                            weights=model_in.layers[0].get_weights(), trainable=False)
        model.add(input_layer)
        for i in xrange(1, nlayers-1):
            curr_layer = Dense(model_in.layers[i].output_dim, init='normal', activation='relu',
                               weights=model_in.layers[i].get_weights(), trainable=False)
            model.add(curr_layer)

        # the final 'layer' will be trained by this class itself, so we compile the model as is
        model.compile(loss='mean_squared_error', optimizer='adam')

        self.model = model
        self.reg_val = 10.0
        self.weights = []

    def transform(self, data):
        # map data manually
        mapped_data = data
        nlayers = len(self.model.layers)
        for i in xrange(nlayers):
            curr_layer = self.model.layers[i]
            mapped_data = curr_layer.activation(np.dot(mapped_data, curr_layer.get_weights()[0])
                                                + curr_layer.get_weights()[1])

        return mapped_data

    def fit(self, data, obs):
        mapped_data = self.transform(data)
        self.weights = solve_tikhinov(mapped_data, obs, reg_val=self.reg_val)

        return self

    def predict(self, data):
        mapped_data = self.transform(data)
        return np.dot(mapped_data, self.weights)


class RBFNetwork(object):
    """
     This class implements a radial basis function network. This is just a single-layer neural
     network/kernel machine with a fixed class of basis functions defining a feature map, while
     the weights generating the function are learned using least squares or gradient descent.
     We go one step further and allow the parameters of the network (other than basis location)
     to be learned from the data, if the user chooses.
    """
    def __init__(self, centers, k_func, params, noise, optimizer=None):
        """
        :param centers:
        :param k_func:
        :param params:
        :param noise:
        :param optimizer:
        """
        if not isinstance(centers, np.ndarray):
            raise InvalidFeatureMapInput("Centers must be a numpy array.")
        if not isinstance(k_func, basestring):
            raise InvalidFeatureMapInput("k_func must be a string.")
        accepted_kernels = ["gaussian", "polynomial", "sqexp"]
        if k_func not in accepted_kernels:
            raise InvalidFeatureMapInput("k_func must be one of the following: " + str(accepted_kernels))
        if not isinstance(params, np.ndarray):
            if params.ndim != 1:
                raise InvalidFeatureMapInput("params must be a 1D numpy array.")
        if not isinstance(noise, float):
            raise InvalidFeatureMapInput("noise must be a float.")
        if noise < 0:
            raise InvalidFeatureMapInput("noise must be a nonnegative float.")
        if optimizer is not None and not isinstance(optimizer, dict):
            raise InvalidFeatureMapInput("optimizer should either be None or a dict.")
        self.centers = centers
        self.nbases = centers.shape[1]
        self.weights = np.random.randn(self.nbases, 1)
        self.k_func = k_func
        self.k_params = params
        self.noise = noise
        self.optimizer = optimizer

        self.params_final = []
        self.jitter = 1e-7
        self.nparams = params.shape[0] + 1  # add noise parameter
        self.k_type = KernelType(self.k_func, self.k_params)

    def transform(self, data, return_grad=False):
        """

        :param data:
        :param return_grad:
        :return:
        """
        self.k_type = KernelType(self.k_func, self.k_params)
        return kernel(self.centers, data, k_type=self.k_type, return_derivs=return_grad)

    def fit(self, data, obs):
        """
        Given a set of data and observations, fit the network with the kernel
        with the current set of parameters.
        :param data: D x nsamp data matrix
        :param obs: 1 x nsamp observation matrix (MUST BE 2D!)
        :return:
        """
        mapped_data = self.transform(data).T
        weights = solve_tikhinov(mapped_data, obs.T, reg_val=pow(self.noise, 2))
        return mapped_data, weights

    def predict(self, data, weights_in=None, return_map=False):
        """
        :param data:
        :param weights_in:
        :param return_map:
        :return:
        """
        kmat = self.transform(data)
        if weights_in is not None:
            if weights_in.ndim == 1:
                weights_in.reshape((weights_in.shape[0], 1))
            fvals = np.dot(weights_in.T, kmat)
        else:
            fvals = np.dot(self.weights.T, kmat)
        if return_map:
            return fvals, kmat
        else:
            return fvals
