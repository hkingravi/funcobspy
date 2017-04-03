"""
This package implements exceptions for all classes.
"""


class InvalidKernelTypeInput(Exception):
    """Exception thrown when incorrect inputs are passed into KernelType class."""
    pass


class InvalidKernelInput(Exception):
    """Exception thrown when incorrect inputs are passed into kernel function."""
    pass


class InvalidSolveTikhinovInput(Exception):
    """Exception thrown when incorrect inputs are passed into solve_tikhinov function."""
    pass


class InvalidFeatureMapInput(Exception):
    """Exception thrown when incorrect inputs are passed into FeatureMap class."""
    pass


class InvalidRBFNetworkInput(Exception):
    """Exception thrown when incorrect inputs are passed into RBFNetwork class."""
    pass

