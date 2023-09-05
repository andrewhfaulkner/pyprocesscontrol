# This module will contain all of the code for process statistics.
# This includes checks for normality, capability, and other statistics

import pandas as pd
import numpy as np
import scipy as sp

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
