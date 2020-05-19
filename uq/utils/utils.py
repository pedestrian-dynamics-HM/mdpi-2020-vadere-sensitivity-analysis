import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from datetime import datetime
import logging
from scipy import stats
from pandas.core.frame import DataFrame
from typing import Union

from uq.utils.datatype import is_vector, is_scalar
from typing import Union

""" ----------------------------------- check inputs --------------------------------------------------- """


def check_inputs(bool_surrogate_data_misfit: bool, bool_surrogate: bool, nr_points_surrogate: int,
                 surrogate_fit_type: str,
                 nr_points_averaged_surrogate: int, method: str, burn_in: int, bool_load_surrogate: bool,
                 run_local: bool):
    if bool_surrogate_data_misfit and bool_surrogate:
        raise Exception(
            "Surrogate can be constructed either for Vadere (bool_surrogate = True) or for the misfit function (bool_surrogate_data_misfit = True) but not for both at the same time")

    if (bool_surrogate_data_misfit or bool_surrogate) and surrogate_fit_type is "spline" and nr_points_surrogate <= 3:
        raise Exception("At least 4 data points are needed for spline interpolation")

    if nr_points_averaged_surrogate < 1:
        raise Exception("Nr Points Averaged Surrogate must be at least 1 (no averaging)")

    if method == "ABC" and burn_in > 0:
        # there is burn-in for ABC with rejection sampling
        # todo: ABC with MCMC has burn in!
        burn_in = 0

    if bool_surrogate_data_misfit and bool_load_surrogate and not run_local:
        run_local = True
        logger = logging.getLogger("check_inputs")
        logger.info(
            "Surrogate is loaded from file, local execution is probably faster since it's only necessary for the (synthetical) data point")
        logger.info("Run_local is set to True -> Vadere runs are performed locally")

    return burn_in, run_local
