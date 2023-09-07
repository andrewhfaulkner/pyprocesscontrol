# This module will contain all of the code for process statistics.
# This includes checks for normality, capability, and other statistics

import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats

import pyprocesscontrol as ppc


def capability(data: ppc.tools.datastructures) -> float:
    """
    The purpose of this function is to calculate the capability statistics of a given dataset/specification

    Args:
        data: pyprocess control datastructures object containg the data and spec limits for the process

    Returns:
        floating point number representing the process capability statistic
    """

    # This function requires some sort of metadata attached to the datastructures object. I will have to add that functionality first.

    pass


def shapiro_wilk(data: pd.Series) -> float:
    """
    This function applies the Shapiro-Wilks test for normality on a pandas series
    Args:
        data: The data to perform the test on
    Returns:
        floating point representing the p-value the distribtution comes from a normally distributed distribution
    """

    try:
        if isinstance(data, pd.Series) != True:
            raise TypeError("Incorrect Type")

    except TypeError:
        data = pd.Series(data)

    statsistic, p_val = stats.shapiro(data)

    return p_val


def cpk(data: pd.Series, lsl: float, usl: float) -> float:
    """
    This function calculates the capability statistic for the dataset given a high and low spec limit.

    Args:
        data: pandas series containing sample data
        lsl: lower spec limit for the data set
        usl: upper spec limit for the data set

    Returns:
        floating point representing the process capability of the system
    """

    pass


def cp(data: pd.Series, lsl: float, usl: float) -> float:
    """
    This function calculates the capability statistic for the dataset given a high and low spec limit.

    Args:
        data: pandas series containing sample data
        lsl: lower spec limit for the data set
        usl: upper spec limit for the data set

    Returns:
        floating point representing the process capability of the system
    """

    pass


def ppk(data: pd.Series, lsl: float, usl: float) -> float:
    """
    This function calculates the capability statistic for the dataset given a high and low spec limit.

    Args:
        data: pandas series containing sample data
        lsl: lower spec limit for the data set
        usl: upper spec limit for the data set

    Returns:
        floating point representing the process capability of the system
    """

    pass


def pp(data: pd.Series, lsl: float, usl: float) -> float:
    """
    This function calculates the capability statistic for the dataset given a high and low spec limit.

    Args:
        data: pandas series containing sample data
        lsl: lower spec limit for the data set
        usl: upper spec limit for the data set

    Returns:
        floating point representing the process capability of the system
    """

    pass
