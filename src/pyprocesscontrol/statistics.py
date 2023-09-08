# This module will contain all of the code for process statistics.
# This includes checks for normality, capability, and other statistics

import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats

import pyprocesscontrol as ppc
from pyprocesscontrol import tools
from pyprocesscontrol.tools import datastructures


def shapiro_wilk(data: pd.Series) -> float:
    """
    This function applies the Shapiro-Wilks test for normality on a pandas series
    Args:
        data: The data to perform the test on
    Returns:
        floating point representing the p-value the samples comes from a normally distributed distribution
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

    stats = data.describe()
    sig_st = std_st(data)

    cupk = (usl - stats.loc["mean"]) / (3 * sig_st)
    clpk = (stats.loc["mean"] - lsl) / (3 * sig_st)

    final = min(cupk[0], clpk[0])
    return final


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

    stats = data.describe()
    sig_st = std_st(data)

    calc_cp = (usl - lsl) / (6 * sig_st)
    return calc_cp


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

    stats = data.describe()

    pupk = (usl - stats.loc["mean"]) / (3 * stats.loc["std"])
    plpk = (stats.loc["mean"] - lsl) / (3 * stats.loc["std"])

    final = min(pupk, plpk)
    return final


def pp(data: pd.Series, lsl: float, usl: float) -> float:
    """
    This function calculates the capability statistic for the dataset given a high and low spec limit.

    Args:
        data: pandas series containing sample data
        lsl: lower spec limit for the data set
        usl: upper spec limit for the data set

    Returns:
        floating point representing the process capability of the system

    Raises:
        TypeError
    """

    if isinstance(data, pd.Series) != True:
        raise TypeError

    stats = data.describe()

    calc_pp = float((usl - lsl) / (6 * stats.loc["std"]))
    return calc_pp


def std_st(series: pd.Series) -> float:
    """
    This function calculates and returns the short term standard deviation of the data

    Args:
        series: pandas Series containg the data

    Returns:
        floating point representing the short term standard deviation

    """

    try:
        if isinstance(series, pd.Series) != True:
            raise TypeError
    except TypeError as e:
        print(f"Expected type pandas series, not {type(series)}")
        raise

    dropped = series[:-1]
    shifted = series[1:]
    rng = abs(shifted - dropped.values)
    r_bar = 1 / (len(rng) - 1) * rng.sum()

    st_std = r_bar / 1.128
    return st_std
