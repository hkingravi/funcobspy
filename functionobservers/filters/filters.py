"""
Filters that are used to infer dynamics.
"""
import numpy as np


class KalmanFilter(object):
    """
    .. codeauthor:: Hassan A. Kingravi <hkingravi@gmail.com>

    Implements a simple Kalman filter that is used by function observers.

    """
    def __init__(self, P_init, Q, R):
        """
        Constructor for KalmanFilter: initialize filter parameters.

        :param P_init: m x m initial error covariance matrix
        :param Q:  m x m process noise covariance matrix
        :param R: n x n measurement noise covariance matrix
        """
        # process matrices
        self.P_k_prev = P_init
        self.Q = Q
        self.R = R

        # dynamics matrices
        self.A = []  # dynamics
        self.C = []  # measurement matrix
        self.nstates = []  # number of states
        self.m_prev = []  # state vector

    def fit(self, A, C, m_init):
        """
        Fit filter with dynamics and measurement operators, and the
        initial state.

        :param A: dynamics operator: m x m matrix
        :param C: measurement operator: n x m matrix
        :param m_init: current measurement vector
        :return:
        """
        self.A = A
        self.C = C
        self.m_prev = m_init
        self.nstates = self.A.shape[0]

    def predict(self, meas_curr):
        """
        Given a vector of current measurements, filter and correct current state.

        :param meas_curr: current measurement vector
        :return: state estimate, updated error covariance matrix
        """
        # predict equations
        m_k_pred = np.dot(self.A, self.m_prev)  # predicted (a priori) state
        P_k_pred = np.dot(self.A, np.dot(self.P_k_prev, self.A.T)) + self.Q # predicted (a priori) covariance estimate

        # update equations
        v_k = meas_curr - np.dot(self.C, m_k_pred)  # residual signal(innovation)
        S_k = np.dot(self.C, np.dot(P_k_pred, self.C.T)) + self.R   # residual covariance

        # compute Kalman gain
        kg1 = np.dot(P_k_pred, self.C.T)
        kg2 = S_k
        K_k = np.linalg.lstsq(kg2.T, kg1.T)[0].T  # K_k = kg1 / kg2

        m_k = m_k_pred + np.dot(K_k, v_k)  # updated(a posteriori) state
        P_k = np.dot((np.eye(self.nstates) - np.dot(K_k, self.C)), P_k_pred)  # updated(a posteriori) covariance

        self.m_prev = m_k
        self.P_k_prev = P_k
        return m_k, P_k

