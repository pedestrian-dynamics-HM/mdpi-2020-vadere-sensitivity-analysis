import unittest
import numpy as np
from numpy.random import RandomState

from uq.active_subspace.active_subspace_with_gradients import active_subspace_with_gradients
from uq.sensitivity_analysis.config import configure_vadere_sa


class TestSeedActiveSubspaces(unittest.TestCase):
    def unit_disabled(func):
        def wrapper(func):
            func.__test__ = False
            return func

        return wrapper

    @unit_disabled
    def test_same_seed_activity_scores_vadere(self):  # takes too long for pipeline - todo: speed up
        seed = 267673926

        scenario_name = "Liddle_osm_v4_short.scenario"

        bool_gradient = False  # is an analytical gradient available? -> True
        bool_averaged = True
        no_runs_averaged = 3  # average over multiple runs?

        step_size = 0.025  # Number of pedestrians wird um 1 erhöht

        alpha = 1
        k = 1  # desired dimension of subspace +1
        M_boot = 0  # Constantine: Typically between 100 and 10^5
        case = " "

        density_type = "uniform"  # input parameter density

        test_input = np.array([[1], [1.34], [0.26], [180], [1.6], [0.5]])  # legal test input

        # parameter limits
        x_lower = np.array([1.0, 0.5, 0.1, 160, 1.6, 30])  # lower bounds for parameters
        x_upper = np.array([5.0, 2.2, 1.0, 200, 3.0, 70])  # upper bounds for parameters

        # parameters
        key = ["queueWidthLoading", "attributesPedestrian.speedDistributionMean",
               "attributesPedestrian.speedDistributionStandardDeviation",
               "sources.[id==3].spawnNumber", "bottleneck_width",
               "pedPotentialHeight"]  # uncertain parameters

        qoi = "max_density.txt"  # quantity of interest

        # configure setup
        test_model, m, path2tutorial = configure_vadere_sa(run_local=True, scenario_name=scenario_name, key=key,
                                                           qoi=qoi)
        step_size_relative = step_size * (x_upper - x_lower)

        result_gradients1, error_c_gradients1, activity_scores1, __, size_subspace1, __, n_samples1, \
        lambda_eig1, w_active1, __, __, __, __, __, __ = \
            active_subspace_with_gradients(
                test_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size_relative, step_size, case, seed, bool_averaged, no_runs_averaged, False,
                False, False, None)

        # configure setup
        test_model, m, path2tutorial = configure_vadere_sa(run_local=True, scenario_name=scenario_name, key=key,
                                                           qoi=qoi)

        result_gradients2, error_c_gradients2, activity_scores2, __, size_subspace2, __, n_samples2, \
        lambda_eig2, w_active2, __, __, __, __, __, __ = \
            active_subspace_with_gradients(
                test_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size_relative, step_size, case, seed, bool_averaged, no_runs_averaged, False,
                False, False, None)

        np.testing.assert_allclose(activity_scores1, activity_scores2)