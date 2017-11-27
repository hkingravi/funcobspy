"""
Classes and helper functions for constructing a kernel observer.
"""

import numpy as np
from sklearn.utils import check_random_state

from functionobservers.mappers import Mapper, FeatureSpaceGenerator
from functionobservers.filters import KalmanFilter
from functionobservers.log_utils import configure_logger, check_pos_int
logger = configure_logger(level="INFO", name="funcobspy")

JITTER = 1e-4


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
    def __init__(self, mapper, nmeas, meas_op_type, d_filter_params, random_state, verbose=False):
        """

        :param mapper: instance of Mapper object, containing information about the kernel
                       and the finite-dimensional kernel mapping.
        :param nmeas: positive integer: number of measurements for measurement operator.
        :param meas_op_type: type of measurement operator: choose from ['random', 'rational'].
        :param d_filter_params: dictionary of parameters to construct Kalman filter in kernel space.
                                Ideally, it should contain the P_init, Q, and R matrices (see
                                `KalmanFilter` documentation). If the dictionary is a None or
                                empty, sensible defaults will be filled in.
        :param random_state:
        :param verbose: boolean controlling whether output gets printed to the console
        """
        if check_pos_int(nmeas):
            self.nmeas = int(nmeas)
        else:
            logger.error(
                "nmeas must be a positive integer."
            )
            raise ValueError("nmeas must be a positive integer.")
        self.meas_op_type = meas_op_type

        if not isinstance(verbose, bool):
            logger.error(
                "verbose must be a bool."
            )
            raise ValueError("verbose must be a bool.")
        self.verbose = verbose
        if not isinstance(mapper, Mapper):
            logger.error(
                "mapper must be an instance of a subclass of Mapper."
            )
            raise ValueError("mapper must be an instance of a subclass of Mapper.")
        self.mapper = mapper
        self.fspace_generator = FeatureSpaceGenerator(self.mapper, self.verbose)
        self.nbases = mapper.nbases

        # initialize filter
        if not isinstance(d_filter_params, dict):
            logger.error(
                "d_filter_params must be an instance of a dict."
            )
            raise ValueError("d_filter_params must be an instance of a dict.")
        else:
            if 'P_init' not in d_filter_params.keys():
                if self.verbose:
                    logger.info(
                        "P_init not provided: initializing default. See class implementation for details."
                    )
                self.P_init = JITTER*np.eye(self.nbases)
            else:
                self.P_init = d_filter_params['P_init']
            if 'Q' not in d_filter_params.keys():
                if self.verbose:
                    logger.info(
                        "Q not provided: initializing default. See class implementation for details."
                    )
                self.Q = JITTER*np.eye(self.nbases)
            else:
                self.Q = d_filter_params['Q']
            if 'R' not in d_filter_params.keys():
                if self.verbose:
                    logger.info(
                        "R not provided: initializing default. See class implementation for details."
                    )
                self.R = JITTER*np.eye(self.nmeas)
            else:
                self.R = d_filter_params['R']
            self.filter = KalmanFilter(self.P_init, self.Q, self.R)
            self.random_state = check_random_state(random_state)  # make proper RandomState instance

    def fit(self, data, obs, d_mopts):
        """

        :param data: nsamp x dim_in x t numpy matrix or tensor
        :param obs: nsamp x dim_out x t numpy matrix or tensor
        :param d_mopts: dictionary of required matrices for meas_type:
                        'random' requires 'data', an N x D data matrix.
                        'rational' requires 'data', and 'A', an M x M dynamics matrix.
        :return:
        """
        # infer kernel parameters from parameter stream, and set mapper
        self.fspace_generator.fit(data, obs)
        d_params, noise = self.fspace_generator.return_final_params()
        self.mapper.set_kernel_params(d_params)


