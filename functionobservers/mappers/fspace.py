"""
Feature-space associated code.
"""
import numpy as np
from functionobservers.mappers import Mapper, unpack_params_nll
from functionobservers.log_utils import configure_logger
logger = configure_logger(level="INFO", name="funcobspy")


SUPPORTED_MEAS = ['random', 'rational']


class FeatureSpaceGenerator(object):
    """
    Class for training a static feature space from time-series data.
    """
    def __init__(self, mapper, verbose=False):
        """

        :param mapper:
        :param verbose:
        """
        if not isinstance(mapper, Mapper):
            logger.error(
                "mapper must be a subclass of Mapper: halting execution."
            )
            raise ValueError("mapper must be a subclass of Mapper: halting execution.")
        if not isinstance(verbose, bool):
            logger.error(
                "verbose must be a boolean: halting execution."
            )
            raise ValueError("verbose must be a boolean: halting execution.")
        self.mapper = mapper
        self.verbose = verbose
        self.param_stream = None  # parameters per time step
        self.d_params_f = None  # final estimate of parameters
        self.noise_f = None  # final estimate of observation noise

    def fit(self, data, obs, **kwargs):
        """
        Given time-series data, perform feature-space fit. This is done by inferring
        parameters per step, and then using a robust statistic to infer final parameters.

        :param data: nsamp x dim_in x t numpy matrix or tensor
        :param obs: nsamp x dim_out x t numpy matrix or tensor
        :param kwargs: keyword args for feature space generator object
        :return:
        """
        if len(data.shape) != 3 or len(obs.shape) != 3:
            logger.error(
                "Data and observations must have three dimensons."
            )
            raise ValueError("Data and observations must have three dimensons.")
        if data.shape[0] != obs.shape[0]:
            logger.error(
                "Data and observations must have same number of samples."
            )
            raise ValueError("Data and observations must have same number of samples.")
        if data.shape[2] != obs.shape[2]:
            logger.error(
                "Data and observations must have same number of steps."
            )
            raise ValueError("Data and observations must have same number of steps.")
        nsteps = obs.shape[2]
        params = np.zeros((nsteps, self.mapper.nparams))
        for i in range(0, nsteps):
            if self.verbose:
                logger.info(
                    "Fitting step {}...".format(i)
                )
            self.mapper.fit(data[:, :, i], obs[:, :, i], **kwargs)
            params[i, :] = self.mapper.params
        self.param_stream = params

    def return_final_params(self):
        """
        Compute final parameters based on param_stream.

        :return:
        """
        if self.param_stream is not None:
            p_robust = np.median(self.param_stream, axis=0)
            d_p, n = unpack_params_nll(p_robust, k_name=self.mapper.kernel_name)
            self.d_params_f = d_p
            self.noise_f = n
            return d_p, n
        else:
            logger.error(
                "param_stream variable is None: cannot return parameters."
            )
            raise RuntimeError("param_stream variable is None: cannot return parameters.")


def measurement_operator(meas_type, nmeas, mapper, d_mopts, random_state):
    """
    Construct measurement operator in accordance with chosen scheme and
    the Mapper object's transformation properties.

    :param meas_type: type of measurement: choose from:
                       - 'random': randomly select subset from data
                       - 'rational': exploit rational canonical structure of
                                     dynamics operator A.
    :param nmeas: number of measurements to use
    :param mapper: instance of Mapper: choose from 'RBFNetwork' and 'RandomKitchenSinks'.
    :param d_mopts: dictionary of required matrices for meas_type:
                    'random' requires 'data', an N x D data matrix.
                    'rational' requires 'data', and 'A', an M x M dynamics matrix.
    :param random_state: numpy RandomState object used for generating random samples.
    :return: (Kmat, meas_basis, meas_inds): Kmat is an L x M measurement operator,
             meas_basis is an L x D submatrix of data, and meas_inds are the indices
             corresponding to the subset selection.
    """
    if meas_type not in SUPPORTED_MEAS:
        logger.error(
            "meas_type {} not supported: choose from {}.".format(meas_type, SUPPORTED_MEAS)
        )
        raise ValueError("meas_type {} not supported: choose from {}.".format(meas_type, SUPPORTED_MEAS))

    if meas_type == "random":
        rand_inds = random_state.permutation(d_mopts['data'].shape[0])
        meas_inds = rand_inds[0:nmeas]
        meas_basis = d_mopts['data'][meas_inds, :]

        # sort basis if necessary
        if mapper.sort_mat:
            s_inds = np.argsort(meas_basis[:, 0])
            meas_basis[:, 0] = meas_basis[s_inds, 0]
            meas_inds = meas_inds[s_inds]
        Kmat = mapper.transform(meas_basis)

    elif meas_type == "rational":
        logger.error(
            "meas_type {} not currently implemented.".format(meas_type)
        )
        raise NotImplementedError("meas_type {} not currently implemented.".format(meas_type))
    return Kmat, meas_basis, meas_inds
