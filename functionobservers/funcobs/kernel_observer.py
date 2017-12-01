"""
Classes and helper functions for constructing a kernel observer.
"""

import numpy as np
from sklearn.utils import check_random_state

from functionobservers.mappers import Mapper, FeatureSpaceGenerator, measurement_operator
from functionobservers.filters import KalmanFilter
from functionobservers.log_utils import configure_logger, check_pos_int, check_pos_float
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
    def __init__(self, mapper, nmeas, meas_op_type, d_filter_params, random_state, reg_parm=0.01, verbose=False):
        """

        :param mapper: instance of Mapper object, containing information about the kernel
                       and the finite-dimensional kernel mapping.
        :param nmeas: positive integer: number of measurements for measurement operator.
        :param meas_op_type: type of measurement operator: choose from ['random', 'rational'].
        :param d_filter_params: dictionary of parameters to construct Kalman filter in kernel space.
                                Ideally, it should contain the P_init, Q, and R matrices (see
                                `KalmanFilter` documentation). If the dictionary is a None or
                                empty, sensible defaults will be filled in.
        :param random_state: random seed, RandomState object, or None
        :param reg_parm: regularization parameter for solving for dynamics operator
        :param verbose: boolean controlling whether output gets printed to the console
        """
        if check_pos_int(nmeas):
            self.nmeas = int(nmeas)
        else:
            logger.error(
                "nmeas must be a positive integer."
            )
            raise ValueError("nmeas must be a positive integer.")
        if check_pos_float(reg_parm):
            self.reg_parm = float(reg_parm)
        else:
            logger.error(
                "reg_parm must be a positive float."
            )
            raise ValueError("reg_parm must be a positive float.")

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
            self.random_state = random_state  #check_random_state(random_state)  # make proper RandomState instance
            self.curr_weights = None

            # filter-specific variables:
            self.filter = KalmanFilter(self.P_init, self.Q, self.R)
            self.meas_inds = None
            self.meas_basis = None

    def fit(self, data, obs, d_mopts, **kwargs):
        """

        :param data: nsamp x dim_in x t numpy matrix or tensor
        :param obs: nsamp x dim_out x t numpy matrix or tensor
        :param d_mopts: dictionary of required matrices for meas_type:
                        'random' requires 'data', an N x D data matrix.
                        'rational' requires 'data', and 'A', an M x M dynamics matrix.
        :param kwargs: keyword args for mapper fitting
        :return:
        """
        if self.verbose:
            logger.info(
                "Inferring kernel parameters from data and observations of shapes {} and {} "
                "respectively.".format(data.shape, obs.shape)
            )
        # infer kernel parameters from parameter stream, and set mapper
        param_stream = self.fspace_generator.fit(data, obs, **kwargs)
        d_params, noise = self.fspace_generator.return_final_params()
        self.mapper.set_params(d_params, noise)

        # infer weights per time step using optimal parameters
        nsteps = data.shape[2]
        weights = np.zeros((nsteps, self.mapper.nbases))
        for i in range(0, nsteps):
            w, _ = self.mapper.fit_current(data[:, :, i], obs[:, :, i])
            weights[i, :] = w

        # infer dynamics operator using least squares
        weight_set = np.vstack((weights[0:nsteps-1, :], weights[nsteps-1, :].reshape((1, self.mapper.nbases))))
        Ai = np.dot(weight_set.T, weight_set) + self.reg_parm*np.eye(self.mapper.nbases)
        Bi = np.dot(weight_set.T, weights)
        A = np.linalg.solve(Ai, Bi)

        # get measurement operator
        Kmat, meas_basis, meas_inds = measurement_operator(self.meas_op_type, self.nmeas, self.mapper, d_mopts,
                                                           random_state=self.random_state)

        # initialize kernel Kalman filter: programmed in column space
        self.curr_weights = weights[-1, :].reshape((1, self.mapper.nbases))  # get current state
        self.filter.fit(A, Kmat.T, m_init=self.curr_weights.T)
        self.meas_inds = meas_inds  # indices for measurement
        self.meas_basis = meas_basis

        return param_stream, weights, meas_inds

    def update(self, meas_te):
        """
        Given a set of measurements, utilize the filter to correct the state.

        :param meas_te: 1 x nmeas prediction numpy array
        :return:
        """
        w, _ = self.filter.predict(meas_curr=meas_te.T)
        self.curr_weights = w.T

    def predict(self, data_te):
        """
        Given data locations, given the current state of the weights, make a prediction.

        :param data_te: Ntest x M test data numpy array
        :return:
        """
        f, K = self.mapper.predict(data_te, weights_in=self.curr_weights)
        return f, K.T
