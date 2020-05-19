import unittest
import numpy as np
from uq.utils.model_function import QuadraticModel


class TestQuadraticFunction(unittest.TestCase):

    def test_gradient(self):
        a = np.random.rand(1)
        b = np.random.rand(1)
        fct = QuadraticModel(a, b)

        var = 10.0
        value_rand = np.array([np.random.rand(1)*var-var/2])

        true_diff = fct.eval_gradient(value_rand)

        h0 = 4e-3
        h1 = 3e-4
        h2 = 2e-5

        approx_diff_h0 = fct.approximate_gradient(value_rand, h0)
        approx_diff_h1 = fct.approximate_gradient(value_rand, h1)
        approx_diff_h2 = fct.approximate_gradient(value_rand, h2)

        error_h0 = np.abs(approx_diff_h0-true_diff)
        error_h1 = np.abs(approx_diff_h1-true_diff)
        error_h2 = np.abs(approx_diff_h2-true_diff)

        self.assertLessEqual(error_h0, np.power(h0, 2))
        self.assertLessEqual(error_h1, np.power(h1, 2))
        self.assertLessEqual(error_h2, np.power(h2, 2))


if __name__ == '__main__':
    unittest.main()