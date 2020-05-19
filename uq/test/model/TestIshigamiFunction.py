import unittest
import numpy as np
import numpy.testing as nptest

from uq.utils.model_function import IshigamiFunction


class TestIshigamiFunction(unittest.TestCase):

    def test_ishigami_function_values(self):
        model = IshigamiFunction(a=7, b=0.05)
        x_lower = np.array([-np.pi, -np.pi, -np.pi])  # https://uqworld.org/t/ishigami-function/55
        x_upper = np.array([np.pi, np.pi, np.pi])  # https://uqworld.org/t/ishigami-function/55

        y_lower = model.eval_model(x_lower)
        y_upper = model.eval_model(x_upper)

        nptest.assert_almost_equal(y_lower, 0)
        nptest.assert_almost_equal(y_upper, 0)


if __name__ == '__main__':
    unittest.main()
