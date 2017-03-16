"""
Utilities for dealing with keras layers, particularly the regression case.
"""
import numpy as np
from keras.layers import Dense


def map_data(model, data):
    """
    Given a DNN keras model with dense layers all the way through, map the data until
    the penultimate layer, before the final summation.

    :param model: keras DNN model with Dense layers.
    :param data: data of shape (N, D).
    :return:
    """
    mapped_data = data
    nlayers = len(model.layers())
    for i in xrange(nlayers-1):
        curr_layer = model.layers()[i]
        mapped_data = curr_layer.activation(np.dot(mapped_data, curr_layer.get_weights()[0])
                                            + curr_layer.get_weights()[1])
    return mapped_data
