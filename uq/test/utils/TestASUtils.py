import unittest
import numpy.testing as nptest
import numpy as np
from numpy.random import RandomState

from nose.tools import assert_raises

from uq.active_subspace.utils import transform_coordinates_from_unit, transform_coordinates_to_unit, \
    find_largest_gap_log, is_smaller_equal_than_machine_precision
from uq.active_subspace.config_matrix_constantine_fct import config_matrix_constantine


class TestUtils(unittest.TestCase):
    def test_transformation_from_to_unit(self):
        print("test_transformation_to_unit")
        seed = 3476868
        case = 1
        test_model, x_lower, x_upper, m, density_type, test_input = \
            config_matrix_constantine(case, random_state=RandomState(seed))

        # test of transformation
        test_x = transform_coordinates_to_unit(x_lower, x_upper, test_input)
        test_y = transform_coordinates_from_unit(x_lower, x_upper, test_x)

        nptest.assert_almost_equal(test_y, np.expand_dims(test_input, axis=1))

    def test_find_spectral_gap(self):
        eigenvalues = np.array([10, 1, 0.8, 0.6, 0.5, 0.1])
        idx, __ = find_largest_gap_log(eigenvalues)
        self.assertEqual(idx, 0)

        eigenvalues = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 9])
        idx, __ = find_largest_gap_log(eigenvalues)
        self.assertEqual(idx, 8)

        eigenvalues = np.array([10, 10, 10, 10, 9.9, 9.9, 9.9, 9.9, 9.9])
        idx, __ = find_largest_gap_log(eigenvalues)
        self.assertEqual(idx, 3)

        eigenvalues = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        idx, __ = find_largest_gap_log(eigenvalues)
        self.assertEqual(idx, 0)

    def test_find_spectral_gap_above_step_size(self):
        # make sure the gap above the step size is found

        # step_size below all values, same gap should be found
        eigenvalues = np.array([10, 1, 0.8, 0.6, 0.5, 0.1])
        idx1, __ = find_largest_gap_log(eigenvalues, step_size=1e-2)
        idx2, __ = find_largest_gap_log(eigenvalues, step_size=None)

        self.assertEqual(idx1, idx2)

        # among equidistantly placed values
        eigenvalues = np.array([1e5, 1e4, 1e3, 1e2, 1e1, 1e-1])
        idx, __ = find_largest_gap_log(eigenvalues, step_size=2e1)
        self.assertEqual(idx, 0)

        # among equidistantly placed above the step size
        eigenvalues = np.array([1e5, 1e5, 1e5, 1e-3, 1e-3, 1e-3])
        idx, __ = find_largest_gap_log(eigenvalues, step_size = 1e-3)
        self.assertEqual(idx, 0)


    def test_find_spectral_gap_duplicate_values(self):
        eigenvalues = np.array([1e-1, 1e-1, 2e-3, 1.9e-3, 1.8e-3, 1.8e-3, 1.7e-3, 1.6e-3])
        idx, __ = find_largest_gap_log(eigenvalues)
        self.assertEqual(idx, 1)

    def test_find_spectral_gap_mixed_vector(self):
        # vector is not ascending
        eigenvalues = np.array([1e-3, 1e-1, 2e-3, 1.9e-3, 1.8e-3, 1.8e-3, 1.7e-3, 1.6e-3])
        assert_raises(ValueError, find_largest_gap_log, eigenvalues)

    def test_is_smaller_than_machine_precision(self):
        self.assertFalse(is_smaller_equal_than_machine_precision(1))
        self.assertTrue(is_smaller_equal_than_machine_precision(np.finfo(float).eps))
        self.assertTrue(is_smaller_equal_than_machine_precision(np.finfo(float).eps - np.finfo(float).eps / 10))
        self.assertFalse(is_smaller_equal_than_machine_precision(np.finfo(float).eps + 0.1))
        self.assertFalse(is_smaller_equal_than_machine_precision(-1))


if __name__ == '__main__':
    unittest.main()
