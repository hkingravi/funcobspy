"""

"""
import numpy as np


def solve_chol(L, B):
    """
    Solve linear system given a Cholesky factorization.
    X = L\(L'\B);
    We solve X = A\B: here, A must be a positive-definite
    N x N matrix, and B must be a 2D array that is N x M.
    The user must pass in L such that X = np.dot(L, L.T).

    :param L: Cholesky factorization of A, i.e. A = LL'
    :param B: linear system to solve, i.e. X = A\B
    :return: X
    """
    return np.linalg.lstsq(L, np.linalg.lstsq(L.T, B)[0])[0]


def solve_tikhinov(A, Y, reg_val):
    """
    Solve reqularized least squares system AX = Y.

    :param A: N x D1 data matrix
    :param Y: N x D2 output matrix
    :param reg_val: nonnegative scalar
    :return:
    """
    if not isinstance(reg_val, float):
        raise InvalidSolveTikhinovInput("reg_val must be a float.")
    if reg_val < 0:
        raise InvalidSolveTikhinovInput("reg_val must be nonnegative.")
    jitter = 1e-7
    ncent = A.shape[1]
    Pmat = np.dot(A.T, A) + reg_val*jitter*np.eye(ncent)
    L = np.linalg.cholesky(Pmat).T
    Ymat = np.dot(A.T, Y)
    return solve_chol(L, Ymat)


class InvalidSolveTikhinovInput(Exception):
    """Exception thrown when incorrect inputs are passed into solve_tikhinov function."""
    pass

