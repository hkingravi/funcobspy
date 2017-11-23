"""
Utilities for loading and storing data, as well as generating synthetic time series.
"""

import numpy as np


def time_varying_uncertainty(weights_star, t, scheme):
    """
    Generate a series of weights from a nonlinear dynamical system.
    These weights will then be used as a means of generating time-varying
    functions for a linear function generator class such as an RBFNetwork.
    For simplicity, these weights are assumed to be 5-dimensional now.


    :param weights_star:
    :param t:
    :param scheme:
    :return:
    """
    if scheme == "smooth1":
        weights_star[0] = 0.999*weights_star[0] + 0.5*np.sin(t)*weights_star[1] + 0.5*np.sin(t)
        weights_star[1] = 0.3*np.cos(1.1*t) + 0.1*weights_star[1]
        vals = np.min(np.array([0.1*np.cos(1.1*t) + weights_star[2], 2]))
        weights_star[2] = np.max(np.array([vals, -2]))
        vals = np.max(np.array([np.min(np.array([0.1*np.cos(1.1*t) + 0.3*np.sin(25*t), 2])), -2]))
        weights_star[3] = np.sin(weights_star[4]) + weights_star[3]*vals
        vals = np.max(np.array([np.min(np.array([0.1*np.cos(1.1*t)*weights_star[0] + weights_star[4], 2])), -2]))
        weights_star[4] = 0.1*np.cos(2.2*t) + vals
    elif scheme == "smooth2":
        weights_star[0] += 0.1 * np.sin(t)
        weights_star[1] = np.cos(1.1 * t) + weights_star[1]
        weights_star[2] = np.max(np.min(0.1 * np.cos(1.1 * t) + weights_star[2], 2), -2)
        weights_star[3] = np.max(np.min(0.1 * np.cos(1.1 * t) + 0.3 * np.sin(25 * t), 2), -2)
        weights_star[4] = np.cos(2.2 * t) + 0.01 * weights_star[4]
    elif scheme == "smooth3":
        weights_star[0] = 0.1 * weights_star[0] + 0.2 * np.sin[t] * weights_star[1] + 0.2 * np.sin[t]
        weights_star[1] = 0.4 * np.cos(1.1 * t) + 0.5 * weights_star[1]
        weights_star[2] = np.max(np.min(0.1 * np.cos(1.1 * t) + weights_star[2], 2), -2)
        weights_star[3] = 0.3 * (np.sin(weights_star[4]) + weights_star[3] * np.max(np.min(0.1 * np.cos(1.1 * t) + 0.3 * np.sin(25 * t), 2), -2))
        weights_star[4] = 0.8 * np.cos(2.2 * t) + np.max(np.min(0.1 * np.cos(1.1 * t) * weights_star[0] + weights_star[4], 2), -2)
    elif scheme == "smooth4":
        weights_star[0] = 0.7 * weights_star[0] + 0.2 * np.sin[t] * weights_star[1] + 0.3 * np.sin[t]
        weights_star[1] = 0.3 * np.cos(1.1 * t) + 0.1 * weights_star[1]
        weights_star[2] = np.max(np.min(0.1 * np.cos(1.1 * t) + weights_star[2], 2), -2)
        weights_star[3] = np.sin(weights_star[4]) + weights_star[3] * np.max(np.min(0.1 * np.cos(1.1 * t) + 0.3 * np.sin(25 * t), 2), -2)
        weights_star[4] = 0.1 * np.cos(2.2 * t) + np.max(np.min(0.1 * np.cos(1.1 * t) * weights_star[0] + weights_star[4], 2), -2)
    elif scheme == "switching":
        if t < 5:
            weights_star[0] = 0.999 * weights_star[0] + 0.001 * np.sin(t)
            weights_star[1] = 0.1 * np.cos(1.1 * t) + 0 * weights_star[1]
            weights_star[2] = np.max(np.array([np.min(np.array([0.1 * np.cos(1.1 * t) + weights_star[2], 2])), -2]))
            weights_star[3] = np.max(np.array([np.min(np.array([0.1 * np.cos(1.1 * t) + 0.3 * np.sin(25 * t), 2])), -2]))
            weights_star[4] = 0.01 * np.cos(2.2 * t) + 0.01 * weights_star[4]
        elif 10 > t > 5:
            weights_star[0] = 0.9 * weights_star[0] + 0.001 * np.cos(t)
            weights_star[1] = 0.1 * np.cos(0.5 * t) + 0 * weights_star[1]
            weights_star[2] = np.max(np.array([np.min(np.array([1.0 * np.cos(1.2 * t) + weights_star[2], 1.5])), -1.5]))
            weights_star[3] = np.max(np.array([np.min(np.array([0.1 * np.cos(1.1 * t) + 0.3 * np.sin(30 * t), 1.2])), -1.2]))
            weights_star[4] = 0.1 * np.cos(2.2 * t) + 0.01 * weights_star[4]
        else:
            weights_star[0] = 0.9 * weights_star[0] + 0.01 * np.cos(t)
            weights_star[1] = 0.1 * np.cos(1.0 * t) + 0 * weights_star[1]
            weights_star[2] = np.max(np.array([np.min(np.array([0.4 * np.cos(1.2 * t) + weights_star[2], 1.0])), -1.0]))
            weights_star[3] = np.max(np.array([np.min(np.array([0.2 * np.cos(1.1 * t) + 0.3 * np.sin(28 * t), 2])), -2]))
            weights_star[4] = 0.1 * np.cos(2.0 * t) + 0.01 * weights_star[4]
    elif scheme == "switching2":
        if t < 5:
            weights_star[0] = 0.999 * weights_star[0] + 0.001 * np.sin(t)
            weights_star[1] = 0.1 * np.cos(1.1 * t) + 0 * weights_star[1]
            weights_star[2] = np.max(np.min(0.1 * np.cos(1.1 * t) + weights_star[2], 2), -2)
            weights_star[3] = np.max(np.min(0.1 * np.cos(1.1 * t) * weights_star[0] + 0.3 * np.sin(25 * t), 2), -2)
            weights_star[4] = 0.01 * np.cos(2.2 * t) * weights_star[1] + 0.01 * weights_star[4]
        elif 10 > t > 5:
            weights_star[0] = 0.9 * weights_star[0] + 0.001 * np.cos(t)
            weights_star[1] = 0.1 * np.cos(0.5 * t * weights_star[0]) + 0 * weights_star[1]
            weights_star[2] = np.max(np.min(1 * np.cos(1.2 * t) + weights_star[2], 1.5), -1.5)
            weights_star[3] = np.max(np.min(0.1 * np.cos(1.1 * t * weights_star[2]) + 0.3 * np.sin(30 * t), 1.2), -1.2)
            weights_star[4] = 0.1 * np.cos(2.2 * t) * weights_star[1] + 0.01 * weights_star[4]
        else:
            weights_star[0] = 0.9 * weights_star[0] * weights_star[1] + 0.01 * np.cos(t)
            weights_star[1] = 0.1 * np.cos(1.0 * t) + weights_star[2]
            weights_star[2] = np.max(np.min(0.4 * np.cos(1.2 * t) + weights_star[2], 1.0), -1.0)
            weights_star[3] = weights_star[4] * np.max(np.min(0.2 * np.cos(1.1 * t) + 0.3 * np.sin(28 * t), 2), -2)
            weights_star[4] = 0.1 * np.cos(2.0 * t * weights_star[0]) + 0.01 * weights_star[4]
    else:
        raise RuntimeError("Incorrect scheme. Please see documentation.")

    return weights_star
