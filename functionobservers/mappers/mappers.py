"""
Classes for mapping data from an input domain to a different feature space.
Note that mappers must be trained.
"""
import sys
from abc import ABCMeta, abstractmethod
from sklearn.utils import check_random_state
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from scipy.optimize import minimize

from functionobservers.optimizers.linalg_o import solve_tikhinov
from functionobservers.mappers.kernel import KernelType, kernel
from functionobservers.utils.func_utils import pack_params_nll, unpack_params_nll
from functionobservers.optimizers.likelihood import negative_log_likelihood
from functionobservers.log_utils import configure_logger
logger = configure_logger(level="INFO", name="funcobspy")

SUPPORTED_RBFN_KERNELS = ['gaussian', 'sqexp']


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


class RBFNetwork(Mapper):
    """
    An instance of a linear RBF neural network.

    """
    def __init__(self, centers, kernel_name, d_params, noise, optimizer, d_opt=None, random_state=None,
                 verbose=False):
        """

        :param centers: M x D matrix of M centers
        :param kernel_name: name of kernel: choose from ['gaussian', 'sqexp']
        :param d_params: dictionary of parameters, with the keys being the parameter names
        :param noise: noise
        :param optimizer: string, method in minimize to use, e.g. "L-BFGS-B"
        :param d_opt:
        :param random_state: seed, or random state
        :param verbose: whether to print information to the console or not
        """
        if kernel_name not in SUPPORTED_RBFN_KERNELS:
            out_m = "kernel_name {} not supported: " \
                    "choose one from {}".format(kernel_name, SUPPORTED_RBFN_KERNELS)
            logger.error(out_m)
            raise ValueError(out_m)

        self.centers = centers
        self.kernel_name = kernel_name
        self.d_params = d_params
        self.nparams = len(self.d_params.keys()) + 1
        self.k_type = KernelType(self.kernel_name, params=self.d_params)
        self.noise = noise
        self.optimizer = optimizer
        self.d_opt = d_opt
        self.random_state = check_random_state(random_state)  # make proper RandomState instance

        self.dim = self.centers.shape[1]
        self.ncent = self.centers.shape[0]
        self.weights = self.random_state.randn(1, self.ncent)  # randomly initialize weights
        self.verbose = verbose

    def fit(self, X, y, reinit_params=True):
        """
        Fit method for RBFNetwork: parameters are fitted using the negative log-likelihood.

        :param X: (nsamp, dim) numpy array of data
        :param y: ()
        :param reinit_params:
        :return:
        """
        d_params_i, noise_i = self.init_params()  # reinitialize parameters to break symmetry
        if self.verbose:
            logger.info("Initial (params, noise): ({}, {})".format(self.d_params, self.noise))
            logger.info("Re-initialized (params, noise): ({}, {})".format(d_params_i, noise_i))
        arg_tup = tuple([X, y, self.kernel_name, "RBFNetwork", self.centers, self.verbose])  # tuple for minimize
        params_i = np.log(pack_params_nll(d_params_i, noise_i, self.kernel_name))  # log to avoid numerical issues
        optobj = minimize(fun=negative_log_likelihood, x0=params_i, args=arg_tup, method=self.optimizer,
                          jac=True, options=self.d_opt)  # run optimizer
        params_o = np.exp(optobj.x)  # convert to original form, and unpack
        d_params_o, noise_o = unpack_params_nll(params_o, self.kernel_name)

        # update params, and fit final weights
        self.d_params = d_params_o
        self.noise = noise_o
        self.k_type = KernelType(self.kernel_name, params=self.d_params)

        if self.verbose:
            logger.info("Final (params, noise): ({}, {}). Fitting weights...\n\n".format(self.d_params, self.noise))
        weights, X_t = self.fit_current(X, y)
        self.weights = weights
        return weights, X_t

    def init_params(self):
        """
        Randomly initialize parameters.

        :return: (nparams,) numpy array
        """
        # randomly initialize positive parameters and rescale to be small so as to not cause numerical issues
        params = np.abs(self.random_state.randn(1, self.nparams).reshape((self.nparams,)))/np.sqrt(self.nparams)
        params[-1] /= self.nparams  # make noise parameter even smaller: tends to be sensitive to large values
        return unpack_params_nll(params, self.kernel_name)

    def fit_current(self, X, y):
        """
        Fit weights of model w.r.t. current parameters.

        :param X: N x D numpy array
        :param y: N x 1 labels numpy array
        :return:
        """
        X_t = self.transform(X)  # project data to kernel space, and solve for weights
        weights = solve_tikhinov(X_t.T, y, pow(self.noise, 2)).T
        return weights, X_t

    def transform(self, X, **kwargs):
        """
        Project data to the kernel space: dependent only on the current kernel
        and the centers.

        :param X: N x D numpy array
        :param kwargs: optional arguments that are ignored for this class.
        :return:
        """
        return kernel(data1=self.centers, data2=X, k_type=self.k_type)

    def predict(self, X, weights_in=None):
        """
        Given data, make regression prediction using current weights,
        kernel, and centers.

        :param X: N x D numpy array
        :param weights_in: make a prediction based on weights passed in directly by the user
        :return:
        """
        if weights_in is not None:
            return np.dot(weights_in, self.transform(X)).T
        else:
            return np.dot(self.weights, self.transform(X)).T


