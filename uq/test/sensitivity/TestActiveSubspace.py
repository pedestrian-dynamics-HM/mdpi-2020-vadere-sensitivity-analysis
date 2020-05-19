import unittest
import numpy.testing as nptest
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState

from uq.active_subspace.active_subspace_with_gradients import active_subspace_with_gradients
from uq.active_subspace.config_matrix_constantine_fct import config_matrix_constantine
from uq.utils.model_function import ExponentialModel


class TestActiveSubspace(unittest.TestCase):
    """ Approximate the gradients of the model with a local linear model and check the resulting active subspace """

    # Use many samples to check if the eigenvalues and eigenvalue gap (of the C matrix) approximate the true matrix
    def test_gradients_with_many_samples(self, bool_plot: bool = False):
        seed = 56158435
        print("test_gradients_with_many_samples")
        case = 2
        bool_gradient = True  # is an analytical gradient available? -> True
        step_size = 10 ** -1  # for (finite-difference) approximation of gradients
        alpha = 1e4  # oversampling factor
        k = 6  # desired dimension of subspace +1
        M_boot = 0  # for bootstrapping

        bool_save_data = False
        bool_print = False

        test_model, x_lower, x_upper, m, density_type, test_input = config_matrix_constantine(case)
        path2results = None
        bool_averaged = None
        no_runs_averaged = None

        # True gradients
        max_rel_error_eig, error_c_gradients, activity_scores, \
        true_activity_scores, size_subspace, path_to_files, n_samples, lambda_eig, \
        w_active, test_y, lambda_eig_true, idx_gap, idx_gap_true, distance_subspace, true_distance = \
            active_subspace_with_gradients(
                test_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size, step_size, case,
                seed, bool_averaged, no_runs_averaged, bool_save_data, bool_print, bool_plot, path2results)

        self.assertLessEqual(np.linalg.norm(activity_scores - true_activity_scores),
                             11)  # based upon run on 2020-02-05 (alpha = 1e4)
        self.assertLessEqual(np.max(max_rel_error_eig), 0.5)  # based upon run on 2020-02-05 (alpha = 1e4)

        nptest.assert_allclose(lambda_eig, lambda_eig_true, rtol=1)  # based upon run on 2020-02-05 (alpha = 1e4)
        nptest.assert_equal(idx_gap, idx_gap_true)
        # nptest.assert_allclose(error_c_gradients, np.zeros(shape=np.size(error_c_gradients)), rtol=10)

    def test_gradients_compared_to_constantine_case1(self):
        # use true gradients
        self.gradients_compared_to_constantine_one_case(case=1)

    def test_gradients_compared_to_constantine_case2(self):
        # use true gradients
        self.gradients_compared_to_constantine_one_case(case=2)

    def test_gradients_compared_to_constantine_case3(self):
        # use true gradients
        self.gradients_compared_to_constantine_one_case(case=3)

    def test_approx_gradients_compared_to_constantine_case1_h0(self):
        # approximate gradients
        self.approx_gradients_compared_to_constantine(case=1, step_size=1e-1)

    def test_approx_gradients_compared_to_constantine_case1_h1(self):
        self.approx_gradients_compared_to_constantine(case=1, step_size=1e-3)

    def test_approx_gradients_compared_to_constantine_case1_h2(self):
        self.approx_gradients_compared_to_constantine(case=1, step_size=1e-5)

    def test_approx_gradients_compared_to_constantine_case2_h0(self):
        # approximate gradients
        self.approx_gradients_compared_to_constantine(case=2, step_size=1e-1)

    def test_approx_gradients_compared_to_constantine_case2_h1(self):
        self.approx_gradients_compared_to_constantine(case=2, step_size=1e-3)

    def test_approx_gradients_compared_to_constantine_case2_h2(self):
        self.approx_gradients_compared_to_constantine(case=2, step_size=1e-5)

    def test_approx_gradients_compared_to_constantine_case3_h0(self):
        # approximate gradients
        self.approx_gradients_compared_to_constantine(case=3, step_size=1e-1)

    def test_approx_gradients_compared_to_constantine_case3_h1(self):
        self.approx_gradients_compared_to_constantine(case=3, step_size=1e-3)

    def test_approx_gradients_compared_to_constantine_case3_h2(self):
        self.approx_gradients_compared_to_constantine(case=3, step_size=1e-5)

    # Check the finite difference approximation to gradients when calculating the active subspace
    def approx_gradients_compared_to_constantine(self, case: int, step_size: float = 1e-5, bool_plot=False):
        print("test_approx_gradients_compared_to_constantine: CASE " + str(case))
        seed = 9895489
        rtol_activity_scores = 1
        rtol_eigenvalues = 5
        bool_gradient = False  # is an analytical gradient available? -> True
        alpha = 2  # oversampling factor
        k = 6  # desired dimension of subspace +1
        M_boot = 0  # for bootstrapping

        bool_save_data = False
        bool_print = False

        test_model, x_lower, x_upper, m, density_type, test_input = config_matrix_constantine(case)
        path2results = None  # os.path.abspath("../results")
        bool_averaged = None
        no_runs_averaged = None

        # True gradients
        max_rel_error_eig, error_c_gradients, activity_scores, \
        true_activity_scores, size_subspace, path_to_files, n_samples, lambda_eig, \
        w_active, test_y, lambda_eig_true, idx_gap, idx_gap_true, distance_subspace, true_distance = \
            active_subspace_with_gradients(
                test_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size, step_size, case,
                seed, bool_averaged, no_runs_averaged, bool_save_data, bool_print, bool_plot, path2results)

        # Check size of relative error in eigenvalue
        self.assertLessEqual(np.max(np.abs(max_rel_error_eig)), 1)

        nptest.assert_allclose(lambda_eig_true, lambda_eig, rtol=rtol_eigenvalues)

        if case > 1:
            # Gap testing makes no sense for case 1 since there is no clear gap
            nptest.assert_equal(idx_gap, idx_gap_true)
        # Check relative error in activity scores

        nptest.assert_allclose(activity_scores, true_activity_scores, rtol=rtol_activity_scores)

        if bool_plot:
            # Check eigenvalues, Figure 3.2, d
            self.plot_eigenvalues_threshold(lambda_eig_true, lambda_eig_true, rtol_eigenvalues, case, step_size)
            self.plot_activity_scores_threshold(activity_scores, true_activity_scores, rtol_activity_scores, case,
                                                step_size)

    # Calculate active subspace with the true gradients
    # Quadratic model (constantine-2015, p. 45ff)
    def gradients_compared_to_constantine_one_case(self, case: int = 1, bool_plot: bool = False):
        seed = 354365
        rtol_eigenvalues = 3
        rtol_activity_scores = 1
        print("gradients_compared_to_constantine_one_case:  CASE " + str(case))
        bool_gradient = True  # is an analytical gradient available? -> True
        step_size = None  # for (finite-difference) approximation of gradients
        alpha = 2  # oversampling factor
        k = 6  # desired dimension of subspace +1
        M_boot = 0  # for bootstrapping

        bool_save_data = False
        bool_print = False

        test_model, x_lower, x_upper, m, density_type, test_input = config_matrix_constantine(case)
        path2results = None  # os.path.abspath("../results")
        bool_averaged = None
        no_runs_averaged = None

        # True gradients
        max_rel_error_eig, error_c_gradients, activity_scores, \
        true_activity_scores, size_subspace, path_to_files, n_samples, lambda_eig, \
        w_active, test_y, lambda_eig_true, idx_gap, idx_gap_true, distance_subspace, true_distance = \
            active_subspace_with_gradients(
                test_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size, step_size, case,
                seed, bool_averaged, no_runs_averaged, bool_save_data, bool_print, bool_plot, path2results)

        # Check size of relative error in eigenvalue
        self.assertLessEqual(np.max(np.abs(max_rel_error_eig)), 1)  # based upon run on 2020-02-05 (alpha = 1e4)

        # Check eigenvalues, Figure 3.2, d
        nptest.assert_allclose(lambda_eig, lambda_eig_true,
                               rtol=rtol_eigenvalues)  # based upon run on 2020-02-05 (alpha = 1e4)

        if case > 1:
            # Gap testing makes no sense for case 1 since there is no clear gap
            nptest.assert_equal(idx_gap, idx_gap_true)

        nptest.assert_allclose(activity_scores, true_activity_scores,
                               rtol_activity_scores, case, step_size)  # based upon run on 2020-02-05 (alpha = 1e4)

        if bool_plot:
            self.plot_eigenvalues_threshold(lambda_eig, lambda_eig_true, rtol_eigenvalues, case, step_size)
            self.plot_activity_scores_threshold(activity_scores, true_activity_scores, rtol_activity_scores, case,
                                                step_size)

    def test_activity_scores_testmodel(self, bool_plot: bool = False):

        rtol_eigenvalues = 5
        rtol_activity_scores = 0.5

        testmodel = ExponentialModel()
        density_type = "uniform"
        dim = testmodel.get_dimension()
        x_lower = -1.0 * np.ones(shape=dim)  # lower bounds for parameters
        x_upper = np.ones(shape=dim)  # lower bounds for parameters
        test_input = x_lower
        alpha = 10
        k = 1 + 1
        bool_gradient = True
        M_boot = 0
        step_size = None
        case = None
        seed = 2456354
        bool_averaged = False
        no_runs_averaged = 1
        bool_save_data = False
        bool_print = False
        path2results = None

        # True gradients
        max_rel_error_eig, error_c_gradients, activity_scores, \
        true_activity_scores, size_subspace, path_to_files, n_samples, lambda_eig, \
        w_active, test_y, lambda_eig_true, idx_gap, idx_gap_true, distance_subspace, true_distance = \
            active_subspace_with_gradients(
                testmodel, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size, step_size, case,
                seed, bool_averaged, no_runs_averaged, bool_save_data, bool_print, bool_plot, path2results)

        # Check eigenvalues, Figure 3.2, d
        nptest.assert_allclose(lambda_eig, lambda_eig_true,
                               rtol=rtol_eigenvalues)  # based upon run on 2020-02-05 (alpha = 1e4)

        nptest.assert_equal(idx_gap, idx_gap_true)

        nptest.assert_allclose(activity_scores, true_activity_scores,
                               rtol_activity_scores)

    ### ---------------------------------------------------------- Plot routines

    def plot_eigenvalues_threshold(self, lambda_eig, lambda_eig_true, rtol_eigenvalues: float, case: str,
                                   step_size: float):
        # Show eigenvalues
        plt.figure()
        plt.semilogy(lambda_eig, 'd-', label='Estimated eigenvalues')
        plt.semilogy(lambda_eig_true, 'x-', label='True eigenvalues')
        upper_threshold = lambda_eig_true * (1 + rtol_eigenvalues)
        lower_threshold = lambda_eig_true / (1 + rtol_eigenvalues)
        plt.plot(upper_threshold, ':k')

        plt.plot(lower_threshold, ':k')
        if step_size is None:
            plt.title(r"Eigenvalues, Case {i:d}, rtol: {k:.2f}".format(i=case, k=rtol_eigenvalues))
        else:
            plt.title(r"Eigenvalues, Case {i:d}, step size: {j:.2e}, rtol: {k:.2f}".format(i=case, j=step_size,
                                                                                           k=rtol_eigenvalues))
        plt.legend()
        plt.show()

    def plot_activity_scores_threshold(self, activity_scores, true_activity_scores, rtol_activity_scores: float,
                                       case: str, step_size: float):
        plt.figure()
        plt.semilogy(activity_scores, 'd-', label='Estimated activity scores')
        plt.semilogy(true_activity_scores, 'x-', label='True activity scores')
        upper_threshold = true_activity_scores * (1 + rtol_activity_scores)
        lower_threshold = true_activity_scores / (1 + rtol_activity_scores)
        plt.plot(upper_threshold, ':k')

        plt.plot(lower_threshold, ':k')
        if step_size is None:
            plt.title(r"Activity scores, Case {i:d}, rtol: {k:.2f}".format(i=case, k=rtol_activity_scores))
        else:
            plt.title(r"Activity scores, Case {i:d}, Step size: {j:.2e}, rtol: {k:.2f}".format(i=case, j=step_size,
                                                                                               k=rtol_activity_scores))
        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
