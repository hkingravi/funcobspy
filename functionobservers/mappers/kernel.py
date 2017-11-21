"""
Classes and functions for computing kernel functions.
"""
import sys
import numpy as np

# FIX!!
import logging
logger = logging.getLogger(__name__)
out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
out_hdlr.setLevel(logging.INFO)
logger.addHandler(out_hdlr)
logger.setLevel(logging.INFO)


class KernelType(object):
    """
    Wrapper for kernel type and its parameters, consumed by kernel function.
    """
    def __init__(self, name, params):
        """
        Constructor for KernelType.
        :param name: string indicating kernel type. Choose from
                       {"gaussian", "sqexp", "laplacian", "polynomial"}
        :param params: a dictionary of parameters. Each kernel has its own
                         dictionary, that must conform to its own structure.
                         "gaussian": {"sigma": np.array([1.0])}
                         "laplacian": {"sigma": np.array([1.0])}
                         "sqexp": {"ell1": np.array([0.1], "ell2": np.array([0.2]),
                                   "nu": np.array([1.0])}, for 2D data
                         "polynomial": {"degree": np.array([5]),
                                        "bias": np.array([0.1])}
                         "sigmoid": {"sigma": np.array([0.1])}
                         ""
        """
        accepted_types = ["gaussian", "sqexp", "laplacian", "polynomial", "sigmoid"]
        if name not in accepted_types:
            logger.error("Incorrect type of kernel selected: see documentation.")
            raise ValueError("Incorrect type of kernel selected: see documentation.")
        if not isinstance(params, dict):
            logger.error("k_params must be a dictionary.")
            raise ValueError("k_params must be a dictionary.")

        if name == "polynomial":
            if not {"degree", "bias"}.issubset(set(params.keys())):
                logger.error("Incorrect number of parameters: polynomial kernel needs degree and bias")
                raise ValueError("Incorrect number of parameters: polynomial kernel needs degree and bias")
        elif name == "gaussian":
            if not {"sigma"}.issubset(set(params.keys())):
                logger.error("Incorrect number of parameters: Gaussian kernel needs bandwidth")
                raise ValueError("Incorrect number of parameters: Gaussian kernel needs bandwidth")
        elif not name == "laplacian":
            if {"sigma"}.issubset(set(params.keys())):
                logger.error("Incorrect number of parameters: Laplacian kernel needs bandwidth")
                raise ValueError("Incorrect number of parameters: Laplacian kernel needs bandwidth")
        elif not name == "sigmoid":
            if {"sigma"}.issubset(set(params.keys())):
                logger.error("Incorrect number of parameters: sigmoid kernel needs scaling")
                raise ValueError("Incorrect number of parameters: sigmoid kernel needs scaling")
        elif name == "sqexp":
            nparams = len(params.keys())
            key_list = ["nu"]
            for i in xrange(nparams-1):
                key_list.append("ell" + str(i+1))
            if not set(key_list).issubset(set(params.keys())):
                logger.error("Incorrect number of parameters: see sqexp description in documentation.")
                raise ValueError("Incorrect number of parameters: see sqexp description in documentation.")

        self.name = name
        self.params = params

    def __str__(self):
        """
        This function prints out the KernelType object.
        """
        return "{} kernel with parameters {}".format(self.name, self.params)


def kernel(data1, data2, k_type, return_grads=False):
    """
    Kernel function. Data must be passed in row-wise.

    :param data1:
    :param data2:
    :param k_type:
    :param return_grads: compute derivatives
    :return:
    """
    k_name = k_type.name  # need to check this type later
    k_params = k_type.params

    # if data is one-dimensional, it must be converted
    if data1.ndim == data2.ndim == 1:
        data1 = np.atleast_2d(data1).T
        data2 = np.atleast_2d(data2).T

    # if base mappers, simply call it
    accepted_types = ["gaussian", "sqexp", "laplacian", "polynomial", "sigmoid"]
    if k_name in accepted_types:
        if k_name == "sqexp":
            nells = 0
            for v in k_params.keys():
                if "ell" in v:
                    nells += 1
            if nells != data1.shape[1]:
                out_m = "Incorrect number of parameters: squared exponential " \
                        "kernel must have D+1 parameters, where D is the input dimension."
                logger.error(out_m)
                raise ValueError(out_m)
        k_mat = kernel_base(data1.T, data2.T, k_type, return_grads)
    else:
        logger.error("Invalid kernel type.")
        raise ValueError("Invalid kernel type.")
    return k_mat


def kernel_base(data1, data2, k_type, return_grads):
    """

    :param data1:
    :param data2:
    :param k_type:
    :param return_grads:
    :return:
    """
    if k_type.name == "gaussian" or k_type.name == "laplacian":
        sigma = k_type.params["sigma"]

        if k_type.name == "gaussian":
            s_val = -1.0/(2*pow(sigma, 2))
        elif k_type.name == "laplacian":
            s_val = -1.0/sigma

        if return_grads:
            if k_type.name == "laplacian":
                logger.error("Laplacian kernel currently doesn't have derivatives implemented.")
                raise ValueError("Laplacian kernel currently doesn't have derivatives implemented.")
            dmat = dist_mat(data1=data1, data2=data2)
            k_mat = np.exp(s_val * dmat)
            if k_type.name == "gaussian":
                grads = {"sigma": (1/(pow(sigma, 3)))*np.multiply(k_mat, dmat)}
        else:
            k_mat = dist_mat(data1=data1, data2=data2)
            k_mat = np.exp(s_val*k_mat)
    elif k_type.name == "polynomial":
        if return_grads:
            logger.error("Polynomial kernel currently doesn't have derivatives implemented.")
            raise ValueError("Polynomial kernel currently doesn't have derivatives implemented.")
        degree = k_type.params["degree"]
        bias = k_type.params["bias"]
        d_vals = np.dot(data1.T, data2)
        d_vals = np.power(d_vals, degree)
        bias_mat = bias*np.ones(d_vals.shape)
        k_mat = d_vals + bias_mat
    elif k_type.name == "sqexp":
        if return_grads:
            k_mat, grads = sqexp_kernel(data1=data1, data2=data2, params=k_type.params,
                                        return_grads=return_grads)
        else:
            k_mat = sqexp_kernel(data1=data1, data2=data2, params=k_type.params, return_grads=False)
    else:
        logger.error("Invalid kernel type.")
        raise ValueError("Invalid kernel type.")

    if return_grads:
        return k_mat, grads
    else:
        return k_mat


def dist_mat(data1, data2):
    """
    Compute squared Euclidean distance between each pair of points between data matrices.

    :param data1: D x N matrix, with N samples.
    :param data2: D x M matrix, with M samples.
    :return: N x M distance matrix.
    """
    n = data1.shape[1]
    m = data2.shape[1]
    d = np.dot(data1.T, data2)
    dx = np.reshape(np.sum(np.power(data1, 2), axis=0), (1, n))
    dy = np.reshape(np.sum(np.power(data2, 2), axis=0), (1, m))
    return np.tile(dx.T, (1, m)) + np.tile(dy, (n, 1)) - 2*d


def sqexp_kernel(data1, data2, params, return_grads=False):
    """
    Code for computing the squared exponential kernel, as well as gradient matrices
    with repect to each dimension for the bandwidth :math:`\ell\in\mathbb{R}^D`,
    as well as the signal variance :math:`\nu\in\mathbb{R}`. The kernel is computed as
    :math:`k(x,y):= \nu^2e^{-(x-y)^TC^{-1}(x-y)}`, where :math:`C` is diagonal with
    parameters :math:`\ell_1, \dots, \ell_D`.

    Note that this function can be called independently of kernel, so it can easily
    interface with

    :param data1: D x N matrix with N samples.
    :param data2: D x M matrix, with M samples.
    :param params: D+1 long parameter vector, with 1:D being :math:`\ell_i` and the
                   last element being signal variance :math:`\nu`.
    :param return_grads: flag indicating whether derivatives need to be returned.
    :return:
    """
    ndim = data1.shape[0]
    ells = ["ell" + str(i + 1) for i in range(0, ndim)]
    ell = np.array([params[v] for v in ells])
    sf2 = pow(float(params["nu"]), 2)
    if ndim == 1:
        e_mat = np.diag(1./ell).reshape((1, 1))
    else:
        e_mat = np.diag(1./ell)
    k_mat = dist_mat(data1=np.dot(e_mat, data1), data2=np.dot(e_mat, data2))
    k_mat = sf2*np.exp(-0.5*k_mat)

    if return_grads:
        ngrads = ndim
        grads = {}
        for i in range(0, ngrads):
            grads[ells[i]] = k_mat*dist_mat(data1=data1/float(ell[i]),
                                            data2=data2/float(ell[i]))
        grads["nu"] = 2*k_mat
        return k_mat, grads
    else:
        return k_mat


def map_data_rbfnet(centers, k_type, data, return_grads=False):
    """
    Given a fixed set of centers and a fixed kernel, map given data to the RBF network's
    feature space.

    :param centers: M x D numpy array of centers
    :param k_type:  `KernelType` object
    :param data: N x D numpy array of data
    :param return_grads: boolean indicating whether gradient should be returned
    :return:
    """
    return kernel(data1=data, data2=centers, k_type=k_type, return_grads=return_grads)


def map_data_rks(centers, k_type, data, return_grads=False):
    """
    Given a fixed set of centers and a fixed kernel, map given data to the RBF network's
    feature space.

    :param centers: M x D numpy array of centers
    :param k_type:  `KernelType` object
    :param data: N x D numpy array of data
    :param return_grads: boolean indicating whether gradient should be returned
    :return:
    """
    if k_type.name != "gaussian":
        logger.error("RandomKitchenSinks not implemented for non-Gaussian kernels. Halting execution.")
        raise ValueError("RandomKitchenSinks not implemented for non-Gaussian kernels. Halting execution.")
    centers /= float(np.sqrt(1.0)*k_type.params["sigma"])  # scale centers
    data_trans = np.dot(data, centers.T)
    m_data = np.hstack((np.sin(data_trans), np.cos(data_trans)))/np.sqrt(float(centers.shape[0]))
    if return_grads:
        grd = np.hstack((-np.cos(data_trans), np.sin(data_trans)))/np.sqrt(float(centers.shape[0]))
        return m_data, {"sigma": grd}
    else:
        return m_data


