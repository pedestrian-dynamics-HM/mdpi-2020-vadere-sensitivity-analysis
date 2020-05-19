import unittest
import numpy.testing as nptest

from uq.active_subspace.config_matrix_constantine_fct import config_matrix_constantine
from uq.active_subspace.active_subspace_with_gradients import active_subspace_with_gradients


class TestSeedActiveSubspaces(unittest.TestCase):
    # use the true gradients
    def test_true_gradients(self):
        seed = 254354
        bool_gradient = True  # is an analytical gradient available? -> True
        step_size = None
        alpha = 2  # oversampling factor
        k = 6  # desired dimension of subspace +1
        M_boot = 0  # for bootstrapping
        case = 3

        bool_save_data = False
        bool_print = False
        bool_plot = False

        test_model, x_lower, x_upper, m, density_type, test_input = config_matrix_constantine(case)
        path2results = None  # os.path.abspath("../results")
        bool_averaged = None
        no_runs_averaged = None

        # True gradients
        max_rel_error_eig1, error_c_gradients1, activity_scores1, \
        true_activity_scores, size_subspace1, path_to_files, n_samples1, lambda_eig1, \
        w_active1, test_y1, lambda_eig_true1, idx_gap1, idx_gap_true, distance_subspace1, true_distance = \
            active_subspace_with_gradients(
                test_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size, step_size, case,
                seed, bool_averaged, no_runs_averaged, bool_save_data, bool_print, bool_plot, path2results)

        max_rel_error_eig2, error_c_gradients2, activity_scores2, \
        true_activity_scores, size_subspace2, path_to_files, n_samples2, lambda_eig2, \
        w_active2, test_y2, lambda_eig_true2, idx_gap2, idx_gap_true, distance_subspace2, true_distance = \
            active_subspace_with_gradients(
                test_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size, step_size, case,
                seed, bool_averaged, no_runs_averaged, bool_save_data, bool_print, bool_plot, path2results)

        nptest.assert_array_equal(max_rel_error_eig1, max_rel_error_eig2)
        nptest.assert_array_equal(error_c_gradients1, error_c_gradients2)
        nptest.assert_array_equal(activity_scores1, activity_scores2)
        nptest.assert_array_equal(size_subspace1, size_subspace2)
        nptest.assert_array_equal(n_samples1, n_samples2)
        nptest.assert_array_equal(lambda_eig1, lambda_eig2)
        nptest.assert_array_equal(w_active1, w_active2)
        nptest.assert_array_equal(test_y1, test_y2)
        nptest.assert_array_equal(lambda_eig_true1, lambda_eig_true2)
        nptest.assert_array_equal(idx_gap1, idx_gap2)
        nptest.assert_array_equal(distance_subspace1, distance_subspace2)

    def test_true_gradients_different_seeds(self):
        seed1 = 254354
        seed2 = 558734

        bool_gradient = True  # is an analytical gradient available? -> True
        step_size = None
        alpha = 2  # oversampling factor
        k = 6  # desired dimension of subspace +1
        M_boot = 0  # for bootstrapping
        case = 3

        bool_save_data = False
        bool_print = False
        bool_plot = False

        test_model, x_lower, x_upper, m, density_type, test_input = config_matrix_constantine(case)
        path2results = None  # os.path.abspath("../results")
        bool_averaged = None
        no_runs_averaged = None

        # True gradients
        max_rel_error_eig1, error_c_gradients1, activity_scores1, \
        true_activity_scores, size_subspace1, path_to_files, n_samples1, lambda_eig1, \
        w_active1, test_y1, lambda_eig_true1, idx_gap1, idx_gap_true, distance_subspace1, true_distance = \
            active_subspace_with_gradients(
                test_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size, step_size, case,
                seed1, bool_averaged, no_runs_averaged, bool_save_data, bool_print, bool_plot, path2results)

        max_rel_error_eig2, error_c_gradients2, activity_scores2, \
        true_activity_scores, size_subspace2, path_to_files, n_samples2, lambda_eig2, \
        w_active2, test_y2, lambda_eig_true2, idx_gap2, idx_gap_true, distance_subspace2, true_distance = \
            active_subspace_with_gradients(
                test_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size, step_size, case,
                seed2, bool_averaged, no_runs_averaged, bool_save_data, bool_print, bool_plot, path2results)

        nptest.assert_raises(AssertionError, nptest.assert_array_equal, max_rel_error_eig1, max_rel_error_eig2)
        nptest.assert_raises(AssertionError, nptest.assert_array_equal, error_c_gradients1, error_c_gradients2)
        nptest.assert_raises(AssertionError, nptest.assert_array_equal, activity_scores1, activity_scores2)
        nptest.assert_raises(AssertionError, nptest.assert_array_equal, lambda_eig1, lambda_eig2)
        nptest.assert_raises(AssertionError, nptest.assert_array_equal, w_active1, w_active2)
        if distance_subspace1 is not None:
            nptest.assert_raises(AssertionError, nptest.assert_array_equal, distance_subspace1, distance_subspace2)

    # use the finite difference approximation
    def test_finite_differences(self):
        seed = 254354
        bool_gradient = False  # is an analytical gradient available? -> True
        step_size = 1e-3
        alpha = 2  # oversampling factor
        k = 6  # desired dimension of subspace +1
        M_boot = 0  # for bootstrapping
        case = 3

        bool_save_data = False
        bool_print = False
        bool_plot = False

        test_model, x_lower, x_upper, m, density_type, test_input = config_matrix_constantine(case)
        path2results = None  # os.path.abspath("../results")
        bool_averaged = None
        no_runs_averaged = None

        # True gradients
        max_rel_error_eig1, error_c_gradients1, activity_scores1, \
        true_activity_scores, size_subspace1, path_to_files, n_samples1, lambda_eig1, \
        w_active1, test_y1, lambda_eig_true1, idx_gap1, idx_gap_true, distance_subspace1, true_distance = \
            active_subspace_with_gradients(
                test_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size, step_size, case,
                seed, bool_averaged, no_runs_averaged, bool_save_data, bool_print, bool_plot, path2results)

        max_rel_error_eig2, error_c_gradients2, activity_scores2, \
        true_activity_scores, size_subspace2, path_to_files, n_samples2, lambda_eig2, \
        w_active2, test_y2, lambda_eig_true2, idx_gap2, idx_gap_true, distance_subspace2, true_distance = \
            active_subspace_with_gradients(
                test_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size, step_size, case,
                seed, bool_averaged, no_runs_averaged, bool_save_data, bool_print, bool_plot, path2results)

        nptest.assert_array_equal(max_rel_error_eig1, max_rel_error_eig2)
        nptest.assert_array_equal(error_c_gradients1, error_c_gradients2)
        nptest.assert_array_equal(activity_scores1, activity_scores2)
        nptest.assert_array_equal(size_subspace1, size_subspace2)
        nptest.assert_array_equal(n_samples1, n_samples2)
        nptest.assert_array_equal(lambda_eig1, lambda_eig2)
        nptest.assert_array_equal(w_active1, w_active2)
        nptest.assert_array_equal(test_y1, test_y2)
        nptest.assert_array_equal(lambda_eig_true1, lambda_eig_true2)
        nptest.assert_array_equal(idx_gap1, idx_gap2)
        if distance_subspace1 is not None:
            nptest.assert_array_equal(distance_subspace1, distance_subspace2)

    def test_finite_differences_different_seeds(self):
        seed1 = 3875615
        seed2 = 986465
        bool_gradient = False  # is an analytical gradient available? -> True
        step_size = 1e-3
        alpha = 2  # oversampling factor
        k = 6  # desired dimension of subspace +1
        M_boot = 0  # for bootstrapping
        case = 3

        bool_save_data = False
        bool_print = False
        bool_plot = False

        test_model, x_lower, x_upper, m, density_type, test_input = config_matrix_constantine(case)
        path2results = None  # os.path.abspath("../results")
        bool_averaged = None
        no_runs_averaged = None

        # True gradients
        max_rel_error_eig1, error_c_gradients1, activity_scores1, \
        true_activity_scores, size_subspace1, path_to_files, n_samples1, lambda_eig1, \
        w_active1, test_y1, lambda_eig_true1, idx_gap1, idx_gap_true, distance_subspace1, true_distance = \
            active_subspace_with_gradients(
                test_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size, step_size, case,
                seed1, bool_averaged, no_runs_averaged, bool_save_data, bool_print, bool_plot, path2results)

        max_rel_error_eig2, error_c_gradients2, activity_scores2, \
        true_activity_scores, size_subspace2, path_to_files, n_samples2, lambda_eig2, \
        w_active2, test_y2, lambda_eig_true2, idx_gap2, idx_gap_true, distance_subspace2, true_distance = \
            active_subspace_with_gradients(
                test_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size, step_size, case,
                seed2, bool_averaged, no_runs_averaged, bool_save_data, bool_print, bool_plot, path2results)

        nptest.assert_raises(AssertionError, nptest.assert_array_equal, max_rel_error_eig1, max_rel_error_eig2)
        nptest.assert_raises(AssertionError, nptest.assert_array_equal, error_c_gradients1, error_c_gradients2)
        nptest.assert_raises(AssertionError, nptest.assert_array_equal, activity_scores1, activity_scores2)
        nptest.assert_raises(AssertionError, nptest.assert_array_equal, lambda_eig1, lambda_eig2)
        nptest.assert_raises(AssertionError, nptest.assert_array_equal, w_active1, w_active2)
        if distance_subspace1 is not None:
            nptest.assert_raises(AssertionError, nptest.assert_array_equal, distance_subspace1, distance_subspace2)
