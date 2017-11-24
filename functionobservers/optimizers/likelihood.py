"""
Module containing likelihoods for optimization purposes.
"""
import numpy as np
from functionobservers.mappers.mappers import KernelType
from functionobservers.mappers.kernel import map_data_rbfnet, map_data_rks
from functionobservers.optimizers.linalg_o import solve_chol
from functionobservers.utils.func_utils import pack_params_nll, unpack_params_nll
from functionobservers.log_utils import configure_logger
logger = configure_logger(level="INFO", name="funcobspy")

SUPPORTED_MAPPERS = ["RBFNetwork", "RandomKitchenSinks"]
JITTER = 1e-7


def negative_log_likelihood(param_vec, X, y, k_name, mapper_type, centers, verbose=False):
    """

    :param param_vec:
    :param X:
    :param y:
    :param k_name:
    :param mapper_type: string, choose from "RBFNetwork" and "RandomKitchenSinks"
    :param centers:  M x D matrix of basis centers
    :param verbose: print more information
    :return:
    """
    if mapper_type not in SUPPORTED_MAPPERS:
        out_m = "Mapper type {} not supported: supported mappers are {}.".format(mapper_type, SUPPORTED_MAPPERS)
        logger.error(
            out_m
        )
        raise ValueError(out_m)

    # compute kernel map, and then kernel matrix
    nsamp = X.shape[0]
    dim = param_vec.shape[0]
    param_vec = param_vec.reshape((dim,))  # ensure vector
    if verbose:
        logger.info(
            "param_vec: {}".format(param_vec)
        )
    k_params = np.exp(param_vec[0:dim-1])
    noise = np.exp(2.0*param_vec[-1])  # take exp to avoid negative parameter scaling issues
    d_params, _ = unpack_params_nll(np.hstack([k_params, noise]), k_name)  # 'unpack' vector
    k_type = KernelType(name=k_name, params=d_params)

    if mapper_type == "RBFNetwork":
        X_t, grads = map_data_rbfnet(centers, k_type, X, return_grads=True)
    elif mapper_type == "RandomKitchenSinks":
        X_t, grads = map_data_rks(centers, k_type, X)
    else:
        logger.error(
            "Unexpected mapper_type {}: halting execution".format(mapper_type)
        )
        raise ValueError("Unexpected mapper_type {}: halting execution".format(mapper_type))

    # solve in the primal: compute the capacitance matrix
    Kp = np.dot(X_t.T, X_t)
    ncent = Kp.shape[0]

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

    nll = np.dot(y.T, alpha)/2 + logdet + nsamp*np.log(2*np.pi*sl)/2  # negative log-likelihood

    # compute gradients
    grads_out = {}
    Q = C_inv - np.dot(alpha, alpha.T)
    for v in grads.keys():
        band_mat = np.dot(X_t, grads[v].T) + np.dot(grads[v], X_t.T)
        grads_out[v] = np.sum(np.sum(Q*band_mat))/2
    noise_mat = 2*noise*np.eye(nsamp)
    noise_out = np.sum(np.sum(Q*noise_mat))/2
    if verbose:
        logger.info(
            "Negative log-likelihood: {}\nGrads out: ({}, {})".format(nll, grads_out, noise_out)
        )
    return nll[0][0], pack_params_nll(grads_out, noise_out, k_name)
