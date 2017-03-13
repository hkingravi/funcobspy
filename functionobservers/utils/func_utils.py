"""

"""
import numpy as np


def pack_state(state):
    """
    Given a state in a dynamical system with shape (ndim, 1), flatten it to (ndim,).
    This works as a helper function to remember what you're doing as opposed to explicitly
    reshaping the states.

    :param state:  numpy array of shape (ndim, 1)
    :return: fstate: numpy array of shape (ndim,)
    """
    return np.squeeze(state)


def unpack_state(state):
    """
    Given a packed state in a dynamical system with shape (ndim), unroll it to (ndim, 1).
    This works as a helper function to remember what you're doing as opposed to explicitly
    reshaping the states.

    :param state: numpy array of shape (ndim,)
    :return: fstate: numpy array of shape (ndim, 1)
    """
    return np.reshape(state, (state.shape[0], 1))

