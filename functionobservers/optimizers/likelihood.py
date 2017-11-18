"""
Module containing likelihoods for optimization purposes.
"""
import numpy as np
import functionobservers.mappers.kernel as fkernel
from functionobservers.optimizers.linalg import solve_chol

SUPPORTED_MAPPERS = ["RBFNetwork", "RandomKitchenSinks"]
JITTER = 1e-7


def negative_log_likelihood(param_vec, X, y, k_name, mapper_type, centers):
    """

    :param param_vec:
    :param X:
    :param y:
    :param k_name:
    :param mapper_type: string, choose from "RBFNetwork" and "RandomKitchenSinks"
    :param centers:  M x D matrix of basis centers
    :return:
    """
    if mapper_type not in SUPPORTED_MAPPERS:
        raise ValueError("Mapper type {} not supported: "
                         "supported mappers are {}.".format(mapper_type, SUPPORTED_MAPPERS))

    # compute kernel map, and then kernel matrix
    nsamp = X.shape[0]
    dim = param_vec.shape[1]
    k_params = np.exp(param_vec[0:dim-1])
    noise = np.exp(2.0*param_vec[dim])  # do this to avoid negative parameter scaling issues
    k_type = fkernel.KernelType(name=k_name, params=k_params)

    if mapper_type == "RBFNetwork":
        X_t = fkernel.map_data_rbfnet(centers, k_type, X)
    elif mapper_type == "RandomKitchenSinks":
        X_t = fkernel.map_data_rks(centers, k_type, X)

    # solve in the primal: compute the capacitance matrix
    Kp = np.dot(X_t.T, X_t)
    ncent = Kp.shape[0]
    print ncent

    # compute L matrix, making sure small noise parameters don't lead to numerical instability
    if noise < 1e-6:
        L = np.linalg.cholesky(Kp + (noise + JITTER)*np.eye(ncent)).T
        sl = 1
    else:
        L = np.linalg.cholesky(Kp/noise + np.eye(ncent)).T
        sl = noise

    A_inv = solve_chol(L, np.eye(ncent))
    C_inv = (np.eye(nsamp) - np.dot(X_t, np.dot(A_inv, X_t.T))/sl)/sl
    alpha = np.dot(C_inv, y)  # compute cofficient vector
    logdet = np.sum(np.log(np.diag(L)).ravel())  # compute log determinant

    print alpha.shape

    #nll = obs*alpha/2 + logdet + nsamp*np.log(2*np.pi*sl)/2  # negative log-likelihood
