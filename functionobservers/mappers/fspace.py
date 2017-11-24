"""
Feature-space associated code.
"""
import numpy as np
import matplotlib.pyplot as plt
from functionobservers.mappers import Mapper, unpack_params_nll
from functionobservers.log_utils import configure_logger
logger = configure_logger(level="INFO", name="funcobspy")


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

