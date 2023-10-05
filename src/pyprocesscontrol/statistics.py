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

    if data.isnull().values.all() == True:
        return np.NaN
    elif len(data) < 3:
        return np.NaN

    if data.dtypes == object:
        return np.NaN

    statsistic, p_val = stats.shapiro(data)

    return round(p_val, 2)


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

    if data.dtype == object:
        return np.NaN

    if stats.loc["count"] == 0:
        return np.NaN

    cupk = (usl.values[0] - stats.loc["mean"]) / (3 * sig_st)
    clpk = (stats.loc["mean"] - lsl.values[0]) / (3 * sig_st)

    final = min(cupk, clpk)
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

    calc_cp = (usl.values[0] - lsl.values[0]) / (6 * sig_st)
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

    if data.dtype == object:
        return np.NaN

    stats = data.describe()

    if stats.loc["count"] == 0:
        return np.NaN

    pupk = (usl.values[0] - stats.loc["mean"]) / (3 * stats.loc["std"])
    plpk = (stats.loc["mean"] - lsl.values[0]) / (3 * stats.loc["std"])

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

    if data.dtype != float:
        try:
            data = data.astype(float)
        except ValueError:
            data = pd.to_numeric(data, errors="coerce")
            data = data.astype(float)

    stats = data.describe()

    if stats.loc["count"] == 0:
        return np.NaN

    try:
        std = stats.loc["std"]
    except KeyError:
        return np.NaN

    calc_pp = (usl.values[0] - lsl.values[0]) / (6 * stats.loc["std"])
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
        if series.dtype == object:
            return np.NaN

    except TypeError as e:
        print(f"Expected type pandas series, not {type(series)}")
        raise

    if len(series) <= 2:
        print("short")
        return np.NaN

    try:
        dropped = series[:-1]
        shifted = series[1:]
        rng = abs(shifted - dropped.values)
        r_bar = 1 / (len(rng) - 1) * rng.sum()
    except:
        print("oops")

    st_std = r_bar / 1.128
    return st_std


def defect_probability(data: pd.Series, usl: float, lsl: float) -> float:
    """
    This function calculates the probabilty of producing an out of spec result. The calculation looks for the area under the normal distribution curve that lies beyond the spec limits.

    Args:
        data: pandas series containing the sample data
        usl: upper spec limit for the calculation
        lsl: lower spec limit for the calculation

    Returns:
        floating point number representing the probability of producing an out of spec result
    """

    try:
        if isinstance(data, pd.Series) != True:
            raise TypeError
        if data.dtype == object:
            return np.nan
    except TypeError as e:
        print(f"Expected type pandas series, no {type(data)}")
        raise

    stat = data.describe()
    l_prob = stats.norm.cdf(lsl.values[0], stat.loc["mean"], stat.loc["std"])
    u_prob = 1 - stats.norm.cdf(usl.values[0], stat.loc["mean"], stat.loc["std"])

    return max(l_prob, u_prob)


def ucl(data: pd.Series) -> float:
    """The ucl function calculates the upper control limit for a set of data

    This function calculates the upper control limit for a set of data. The upper control limit is typically 3 standard deviations above the mean of the dataset.

    Parameters
    ----------
    data: pandas.Series
        A pandas series of the data to be analyzed

    Returns
    -------
    float
        Floating point number representing the upper control limit of a dataset

    Notes
    -----

    """

    try:
        if isinstance(data, pd.Series) != True:
            raise TypeError
        if data.dtype == object:
            return np.NaN

    except TypeError as e:
        print(f"Expected type pandas series, not {type(data)}")
        raise

    if len(data) <= 2:
        print("short")
        return np.NaN

    stats = data.describe()

    upper_control = stats.loc["mean"] + 3 * stats.loc["std"]

    return upper_control


def lcl(data: pd.Series) -> float:
    """The lcl function calculates the lower control limit for a set of data

    This function calculates the lower control limit for a set of data. The lower control limit is typically 3 standard deviations below the mean of the dataset.

    Parameters
    ----------
    data: pandas.Series
        A pandas series of the data to be analyzed

    Returns
    -------
    float
        Floating point number representing the lower control limit of a dataset

    Notes
    -----

    """

    try:
        if isinstance(data, pd.Series) != True:
            raise TypeError
        if data.dtype == object:
            return np.NaN

    except TypeError as e:
        print(f"Expected type pandas series, not {type(data)}")
        raise

    if len(data) <= 2:
        print("short")
        return np.NaN

    stats = data.describe()

    upper_control = stats.loc["mean"] - 3 * stats.loc["std"]

    return upper_control
