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


def infer_dynamics(weight_stream, reg_parm):
    """
    Given a stream of weights, infer dynamics operator and covariance matrix.
    Recall that while feature data is in row major form, we perform all dynamics
    operations in column major form to preserve convention and my own sanity.

    :param weight_stream: ndim x T matrix of weights, where T is the number of time
                          steps
    :param weight_stream: regularization parameter
    :return: A, an ndim x ndim dynamics matrix, and B, an ndim x ndim covariance matrix.
    """
    ndim = weight_stream.shape[0]
    nsteps = weight_stream.shape[1]

