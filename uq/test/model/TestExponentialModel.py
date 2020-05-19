import unittest
import numpy as np

from uq.utils.model_function import ExponentialModel
from uq.utils.prior_distribution import UniformGenMult


class TestExponentialModel(unittest.TestCase):

    def test_eval_model(self):
        testmodel = ExponentialModel()
        value2d = np.array([1, 2])
        model_eval_2d = testmodel.eval_model(value2d)

        value3d = np.array([[1, 2]])
        model_eval_3d = testmodel.eval_model(value3d)

        self.assertEqual(model_eval_2d, model_eval_3d)

    def test_approximate_gradient(self):
        testmodel = ExponentialModel()
        value2d = np.array([1, 2])
        value2d = np.random.rand(2)
        testmodel.eval_model(value2d)
        exact_gradient = testmodel.eval_gradient(value2d)
        # testmodel.approximate_gradient(value2d, 0.1, None, 1)
        # testmodel.approximate_gradient(value2d, 0.01, None, 1)

        value3d = np.expand_dims(value2d, axis=1)
        h1 = 1e-1
        h2 = 1e-3
        h3 = 1e-5
        exact_gradient = testmodel.eval_gradient(value3d)
        approximate_gradient_h1 = testmodel.approximate_gradient(value3d, h1, None, 1)
        approximate_gradient_h2 = testmodel.approximate_gradient(value3d, h2, None, 1)
        approximate_gradient_h3 = testmodel.approximate_gradient(value3d, h3, None, 1)

        diff_h1 = np.linalg.norm(approximate_gradient_h1 - exact_gradient)
        diff_h2 = np.linalg.norm(approximate_gradient_h2 - exact_gradient)
        diff_h3 = np.linalg.norm(approximate_gradient_h3 - exact_gradient)

        self.assertLessEqual(diff_h2, diff_h1)  # closer result with smaller step_size (for finite diff.)

        self.assertLessEqual(diff_h3, np.square(h3))  # error smaller than order of truncation error O(\delta x)^2
        self.assertLessEqual(diff_h2, np.square(h2))
        self.assertLessEqual(diff_h1, np.square(h1))

    def test_get_C_matrix(self):
        testmodel = ExponentialModel()
        m = testmodel.get_dimension()
        x_lower = -1.0 * np.ones(shape=m)  # lower bounds for parameters
        x_upper = np.ones(shape=m)  # lower bounds for parameters
        rho = UniformGenMult(lower=x_lower, upper=x_upper, dim=m)
        C_mat1 = testmodel.get_C_matrix(rho)
        C_mat2 = np.array([[0.707222, 0.303095], [0.303095, 0.129898]]) # for U([-1,-1],[1 1]) case

        np.testing.assert_array_almost_equal(C_mat1, C_mat2)


    # todo: finish test

    """def test_active_subspace(self):
        test_model = TestModel()
        density_type = "uniform"
        m = 2
        x_lower = -1.0 * np.ones(shape=m)  # lower bounds for parameters
        x_upper = np.ones(shape=m)  # lower bounds for parameters
        test_input = x_lower
        alpha = 2
        k = 1+1
        bool_gradient = True
        M_boot = 0
        step_size = None
        case = None


        max_rel_error_eig, error_c_gradients, activity_scores, \
        true_activity_scores, size_subspace, path_to_files, n_samples, lambda_eig, \
        w_active, test_y, lambda_eig_true, idx_gap, idx_gap_true, distance_subspace, true_distance = \
            active_subspace_with_gradients(
                test_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size, step_size, case,
                seed, bool_averaged, no_runs_averaged, bool_save_data, bool_print, bool_plot, path2results)"""


if __name__ == '__main__':
    unittest.main()
