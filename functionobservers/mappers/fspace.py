"""
Feature-space associated code.
"""
import numpy as np
from functionobservers.mappers.mappers import Mapper
from functionobservers.log_utils import configure_logger
logger = configure_logger(level="INFO", name="funcobspy")


class FeatureSpaceGenerator(object):
    """
    Class for training a static feature space from time-series data.
    """
    def __init__(self, mapper):
        """

        :param mapper:
        """
        if not isinstance(mapper, Mapper):
            logger.error(
                "mapper must be a subclass of Mapper: halting execution."
            )
            raise ValueError("mapper must be a subclass of Mapper: halting execution.")
        self.mapper = mapper

    def fit(self, data, obs):
        """
        Given time-series data, perform feature-space fit.

        :param data: nsamp x dim_in x t numpy matrix or tensor
        :param obs: nsamp x dim_out x t numpy matrix or tensor
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
        params = np.zeros(nsteps, self.mapper.nparams)
        for i in range(0, nsteps):
            self.mapper.fit(data[:, :, i], obs[:, :, i])
