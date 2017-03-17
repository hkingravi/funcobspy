"""
Classes for mapping data from an input domain to a different feature space.
Note that mappers must be trained.
"""
from abc import ABCMeta, abstractmethod
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
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
