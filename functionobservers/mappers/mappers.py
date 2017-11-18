"""
Classes for mapping data from an input domain to a different feature space.
Note that mappers must be trained.
"""
from abc import ABCMeta, abstractmethod
<<<<<<< HEAD
from sklearn.utils import check_random_state
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from functionobservers.optimizers import solve_tikhinov
from functionobservers.mappers.kernel import KernelType


SUPPORTED_RBFN_KERNELS = ['gaussian', 'sqexp']
=======
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from functionobservers.custom_exceptions import *
from functionobservers.mappers import KernelType, kernel
from functionobservers.solvers import solve_tikhinov
>>>>>>> 34a56d82cbbad7b85363c2023cef306dbb4759c5


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


<<<<<<< HEAD
class RBFNetwork(Mapper):
    """
    An instance of an RBFNetwork.

    """
    def __init__(self, centers, kernel_name, d_params, noise, d_opt, random_state=None):
        """

        :param centers: M x D matrix of M centers
        :param kernel_name: name of kernel: choose from ['gaussian', 'sqexp']
        :param d_params: dictionary of parameters, with the keys being the parameter names
        :param noise:
        :param d_opt:
        :param random_state: seed, or random state
        """
        if kernel_name not in SUPPORTED_RBFN_KERNELS:
            raise ValueError("kernel_name {} not supported: "
                             "choose one from {}".format(kernel_name, SUPPORTED_RBFN_KERNELS))

        self.centers = centers
        self.kernel_name = kernel_name
        self.d_params = d_params
        self.nparams = len(self.d_params.keys()) + 1
        #self.k_type = fkern.KernelType()

        self.noise = noise
        self.d_opt = d_opt
        self.random_state = check_random_state(random_state)

        self.ncent = self.centers.shape[0]
        self.weights = self.random_state.randn(1, self.ncent)

    def fit(self, data, obs, **kwargs):
        """

        :param data:
        :param obs:
        :param kwargs:
        :return:
        """
        if "seed" in kwargs.keys():
            self.seed = kwargs["seed"]
            np.random.seed(kwargs["seed"])
        if "sort_mat" in kwargs.keys():
            self.sort_mat = kwargs["sort_mat"]
        if self.model_type == "RBFNetwork":
            self.nbases = kwargs["centers"].shape[0]
        elif self.model_type == "RandomKitchenSinks":
            self.nbases = kwargs["nbases"]
        else:
            raise RuntimeError("Incorrect model_type {}".format(self.model_type))


    def fit_current(self, X, y):
        """
        Fit weights w.r.t. current parameters.

        :param X: N x D numpy array
        :param y: N x 1 labels numpy array
        :return:
        """
        #X_t = fkern.map_data_rbfnet(self.centers, k_type, X)


    def transform(self, data, **kwargs):
        return data

    def predict(self, data, **kwargs):
        return data
=======
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
>>>>>>> 34a56d82cbbad7b85363c2023cef306dbb4759c5
