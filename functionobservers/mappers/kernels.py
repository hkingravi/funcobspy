"""
Code for kernel learning.
"""

import numpy as np

from functionobservers.custom_exceptions import InvalidKernelInput, InvalidKernelTypeInput


class KernelType(object):
    """
    Wrapper for kernel type and its parameters, consumed by kernel function.
    """
    def __init__(self, k_name, k_params):
        """
        Constructor for KernelType.
        :param k_name: string indicating kernel type. Choose from
                       {"gaussian", "sqexp", "laplacian", "polynomial"}
        :param k_params:
        """
        accepted_types = ["gaussian", "sqexp", "laplacian", "polynomial", "sigmoid"]
        if k_name not in accepted_types:
            raise InvalidKernelTypeInput("Incorrect type of kernel selected: see documentation.")
        if not isinstance(k_params, np.ndarray):
            raise InvalidKernelTypeInput("k_params must be a numpy array.")

        if k_name == "polynomial":
            k_size = k_params.shape
            if k_size[0] != 2:
                raise InvalidKernelTypeInput("Incorrect number of parameters: polynomial kernel needs degree and bias")
        elif k_name == "gaussian":
            k_size = k_params.shape
            if k_size[0] != 1:
                raise InvalidKernelTypeInput("Incorrect number of parameters: Gaussian kernel needs bandwidth")
        elif k_name == "laplacian":
            k_size = k_params.shape
            if k_size[0] != 1:
                raise InvalidKernelTypeInput("Incorrect number of parameters: Laplacian kernel needs bandwidth")
        elif k_name == "sigmoid":
            k_size = k_params.shape
            if k_size[0] != 1:
                raise InvalidKernelTypeInput("Incorrect number of parameters: sigmoid kernel needs scaling")
        # note: squared exponential kernel can have multiple dimensions, so no point checking here

        self.name = k_name
        self.params = k_params

    def __str__(self):
        """
        This function prints out the KernelType object.
        """
        return self.name + " kernel with parameters " + np.array_str(self.params)


def kernel(data1, data2, k_type, return_derivs=False):
    """
    Kernel function. Data must be passed in columnwise.
    :param data1:
    :param data2:
    :param k_type:
    :return:
    """
    k_name = k_type.name # need to check this type later
    k_params = k_type.params

    # if data is one-dimensional, it must be converted
    if data1.ndim == data2.ndim == 1:
        data1 = np.atleast_2d(data1)
        data2 = np.atleast_2d(data2)

    # if base mappers, simply call it
    accepted_types = ["gaussian", "sqexp", "laplacian", "polynomial", "sigmoid"]
    if k_name in accepted_types:
        if k_name == "sqexp":
            if k_params.shape[0] != data1.shape[0] + 1:
                raise InvalidKernelInput("Incorrect number of parameters: squared exponential "
                                         "kernel must have D+1 parameters, where D is the input dimension")
        k_mat = kernel_base(data1, data2, k_type, return_derivs)
    else:
        raise InvalidKernelInput("Invalid kernel type")
    return k_mat


def kernel_base(data1, data2, k_type, return_derivs):
    """
    :param data1:
    :param data2:
    :param k_type:
    :return:
    """
    if k_type.name == "gaussian" or k_type.name == "laplacian":
        sigma = k_type.params[0]

        if k_type.name == "gaussian":
            s_val = -1.0/(2*pow(sigma, 2))
        elif k_type.name == "laplacian":
            s_val = -1.0/sigma

        if return_derivs:
            if k_type.name == "laplacian":
                raise InvalidKernelInput("Laplacian kernel currently doesn't have derivatives implemented.")
            dmat = dist_mat(data1=data1, data2=data2)
            k_mat = np.exp(s_val * dmat)
            if k_type.name == "gaussian":
                derivs_mat = (1/(pow(sigma, 3)))*np.multiply(k_mat, dmat)
        else:
            k_mat = dist_mat(data1=data1, data2=data2)
            k_mat = np.exp(s_val*k_mat)
    elif k_type.name == "polynomial":
        if return_derivs:
            raise InvalidKernelInput("Polynomial kernel currently doesn't have derivatives implemented.")
        degree = k_type.params[0]
        bias = k_type.params[1]
        d_vals = np.dot(data1.T, data2)
        d_vals = np.power(d_vals, degree)
        bias_mat = bias*np.ones(d_vals.shape)
        k_mat = d_vals + bias_mat
    elif k_type.name == "sqexp":
        if return_derivs:
            k_mat, derivs_mat = sqexp_kernel(data1=data1, data2=data2, params=k_type.params,
                                             return_derivs=return_derivs)
        else:
            k_mat = sqexp_kernel(data1=data1, data2=data2, params=k_type.params, return_derivs=return_derivs)
    else:
        raise InvalidKernelInput("Invalid kernel type")

    if return_derivs:
        return k_mat, derivs_mat
    else:
        return k_mat


def dist_mat(data1, data2):
    """
    Compute squared Euclidean distance between each pair of points between data matrices.
    :param data1: D x N matrix, with N samples.
    :param data2: D x M matrix, with M samples.
    :return: M x N distance matrix.
    """
    n = data1.shape[1]
    m = data2.shape[1]
    d = np.dot(data1.T, data2)
    dx = np.reshape(np.sum(np.power(data1, 2), axis=0), (1, n))
    dy = np.reshape(np.sum(np.power(data2, 2), axis=0), (1, m))
    return np.tile(dx.T, (1, m)) + np.tile(dy, (n, 1)) - 2*d


def sqexp_kernel(data1, data2, params, return_derivs=False):
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
    :param return_derivs: flag indicating whether derivatives need to be returned.
    :return:
    """
    ndim = data1.shape[0]
    ell = params[0:ndim]
    sf2 = pow(float(params[-1]), 2)
    k_mat = dist_mat(data1=np.dot(np.diag(1./ell), data1),
                     data2=np.dot(np.diag(1./ell), data2))
    k_mat = sf2*np.exp(-0.5*k_mat)

    if return_derivs:
        nderivs = params.shape[0]-1
        derivs_mat = np.zeros((k_mat.shape[0], k_mat.shape[1], nderivs+1))  # preallocate for efficiency
        for i in range(0, nderivs):
            derivs_mat[:, :, i] = k_mat*dist_mat(data1=data1/float(ell[i]),
                                                 data2=data2/float(ell[i]))
        derivs_mat[:, :, nderivs] = 2*k_mat
        return k_mat, derivs_mat
    else:
        return k_mat
