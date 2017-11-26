"""
Classes and helper functions for constructing a kernel observer.
"""

import numpy as np
from functionobservers.mappers import Mapper, unpack_params_nll
from functionobservers.log_utils import configure_logger
logger = configure_logger(level="INFO", name="funcobspy")


class KernelObserver(object):
    """
    Class for training a kernel observer from time-series data.

    We follow these steps:
    1. Load mapper object for feature space (e.g. RandomKitchenSinks)
    2. Run FeatureSpaceGenerator on time-series data, to get optimal
       feature space parameters (we will, for now, fit a mean to the
       time series)
    3. Map the same time series to using this new feature space object,
       and learn the weights of the regression.
    4. Utilize matrix-valued least squares to infer A operator.
    5. Utilize other methods such a random placement or measurement map
       to construct K operator.
    6. Initialize covariance matrix of weights (or use scaled identity),
       and utilize this matrix to create a kernel Kalman filter.
    7. Now that you have the observer, you can use it for prediction
       etc.
    """
    def __init__(self, mapper, nmeas, meas_op_type, d_filter_params, verbose=False):
        """

        :param mapper: instance of Mapper object, containing information about the kernel
                       and the finite-dimensional kernel mapping.
        :param nmeas: positive integer: number of measurements for measurement operator.
        :param meas_op_type: type of measurement operator: choose from ['random', 'rational'].
        :param d_filter_params: dictionary of parameters to construct Kalman filter in kernel space.
        :param verbose:
        """
