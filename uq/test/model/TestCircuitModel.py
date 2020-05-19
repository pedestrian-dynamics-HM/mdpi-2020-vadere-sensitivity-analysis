import unittest
import numpy as np
import numpy.testing as nptest

from uq.utils.model_function import CircuitModel


class TestCircuitModel(unittest.TestCase):
    def test_approximate_gradient(self):
        circuit_model = CircuitModel()
        value = np.array([60, 30, 2, 1.3, 0.26, 299])

        h1 = 1e-1
        h2 = 1e-3
        h3 = 1e-5
        exact_gradient = circuit_model.eval_gradient(value)
        approximate_gradient_h1 = circuit_model.approximate_gradient(value, h1, None, 1)
        approximate_gradient_h2 = circuit_model.approximate_gradient(value, h2, None, 1)
        approximate_gradient_h3 = circuit_model.approximate_gradient(value, h3, None, 1)

        diff_h1 = np.linalg.norm(approximate_gradient_h1 - exact_gradient)
        diff_h2 = np.linalg.norm(approximate_gradient_h2 - exact_gradient)
        diff_h3 = np.linalg.norm(approximate_gradient_h3 - exact_gradient)

        nptest.assert_array_less(diff_h2, diff_h1)  # closer result with smaller step_size (for finite diff.)
        nptest.assert_array_less(diff_h3, diff_h2)  # closer result with smaller step_size (for finite diff.)

        self.assertLessEqual(diff_h3, np.square(h3))  # error smaller than order of truncation error O(\delta x)^2
        self.assertLessEqual(diff_h2, np.square(h2))
        self.assertLessEqual(diff_h1, np.square(h1))

    def test_eval_model(self):
        circuit_model = CircuitModel()
        value = np.array([60, 30, 2, 1.3, 0.26, 299])

        circuit_model.eval_model(value)

    def test_eval_gradient(self):
        circuit_model = CircuitModel()
        value = np.array([60, 30, 2, 1.3, 0.26, 299])

        circuit_model.eval_gradient(value)



if __name__ == '__main__':
    unittest.main()
