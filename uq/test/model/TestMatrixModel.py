import unittest
import numpy as np
from uq.utils.model_function import MatrixModel

MACHINE_PRECISION = np.finfo(float).eps


class TestMatrixModel(unittest.TestCase):

    def test_eval_model(self):
        m = 10
        eig_A = 10 ** np.linspace(2, -2, m)
        A = np.diag(eig_A)
        matrix_model = MatrixModel(A)

        value_rand = np.random.rand(m)
        eval_nd = matrix_model.eval_model(value_rand)

        value = 1
        eval_0d = matrix_model.eval_model(value)
        eval_1d = matrix_model.eval_model(np.array(1))

        # assure that dimensionality of input does not change output
        self.assertLessEqual(np.linalg.norm(eval_0d - eval_1d), MACHINE_PRECISION)

    def test_gradient_approximation(self):
        m = 10
        eig_A = 10 ** np.linspace(2, -2, m)

        tmp = np.random.rand(m, m)
        Q, R = np.linalg.qr(tmp)
        A = np.matmul(Q, np.matmul(np.diag(eig_A), np.transpose(Q)))

        matrix_model = MatrixModel(A)
        value_rand = np.expand_dims(np.random.rand(m) * 2 - 1, axis=1)

        h0 = 1.0
        h1 = 1e-3
        h2 = 1e-4

        # ground truth
        exact_gradient = matrix_model.eval_gradient(value_rand)

        # approximation
        approx_gradient_h0 = matrix_model.approximate_gradient(value_rand, h0, None, 1)
        approx_gradient_h1 = matrix_model.approximate_gradient(value_rand, h1, None, 1)
        approx_gradient_h2 = matrix_model.approximate_gradient(value_rand, h2, None, 1)

        diff_h0 = np.linalg.norm(approx_gradient_h0 - exact_gradient)
        diff_h1 = np.linalg.norm(approx_gradient_h1 - exact_gradient)
        diff_h2 = np.linalg.norm(approx_gradient_h2 - exact_gradient)

        # asssure that the magnitude of the error fits the method
        self.assertLessEqual(diff_h0, np.square(h0))
        self.assertLessEqual(diff_h1, np.square(h1))
        self.assertLessEqual(diff_h2, np.square(h2))

        # self.assertTrue(diff_h0 - diff_h1 <= np.square(h2))
        # self.assertTrue(diff_h1 - diff_h2 <= np.square(h2))
