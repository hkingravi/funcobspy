"""
Classes for mapping data from an input domain to a different feature space.
Note that mappers must be trained.
"""
from abc import ABCMeta, abstractmethod
from keras.models import Sequential


class Mapper:
    """
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
        print kwargs
        self.model.fit(data, obs, **kwargs)

    def predict(self, data, **kwargs):
        pass

    def score(self, data, labels, **kwargs):
        return self.model.evaluate(data, labels, **kwargs)
