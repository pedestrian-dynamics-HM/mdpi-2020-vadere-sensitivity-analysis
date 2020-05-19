import unittest
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
import chaospy as cp
from numpy.random import RandomState

from uq.utils.model_function import CircuitModel
from uq.active_subspace.utils import relative_error_constantine_2017


class TestSobolChaospy(unittest.TestCase):

    def unit_disabled(func):
        def wrapper(func):
            func.__test__ = False
            return func

        return wrapper

    @unit_disabled
    def test_circuit_model_order_2(self, order_cp: int = 2, bool_plot: bool = False):
        dim = 6
        key = ["R_b1", "R_b2", "R_f", "R_c1", "R_c2", "beta"]
        sobol_indices_quad_constantine = np.array(
            [5.0014515064e-01, 4.1167859899e-01, 7.4006053045e-02, 2.1802568214e-02, 5.1736552010e-08,
             1.4938996627e-05])

        M_constantine = np.array([50, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4])
        sobol_indices_error_constantine = np.transpose(np.array(
            [[6.1114622870e-01, 2.7036543475e-01, 1.5466638009e-01, 1.2812367577e-01, 5.0229955234e-02,
              3.5420048253e-02, 1.4486328386e-02],
             [6.0074404490e-01, 3.2024096457e-01, 1.2296426366e-01, 9.6725945246e-02, 5.3143328175e-02,
              3.2748864016e-02, 1.1486316472e-02],
             [1.1789694228e-01, 4.6150927239e-02, 2.6268692965e-02, 1.8450563871e-02, 8.3656592318e-03,
              5.8550974309e-03, 2.8208921925e-03],
             [3.8013619286e-02, 1.6186288112e-02, 8.9893920304e-03, 6.3911249578e-03, 2.6219049423e-03,
              1.9215077698e-03, 9.5390224479e-04],
             [1.2340746448e-07, 4.8204289233e-08, 3.0780845307e-08, 2.5240466147e-08, 1.0551377101e-08,
              6.9506139894e-09, 3.3372151408e-09],
             [3.4241277775e-05, 1.8074628532e-05, 7.1554659714e-06, 5.0303467614e-06, 2.7593313990e-06,
              1.9529470403e-06, 7.2840043686e-07]]))

        x_lower = np.array([50, 25, 0.5, 1.2, 0.25, 50])  # table 3, constantine-2017
        x_upper = np.array([150, 70, 3.0, 2.5, 1.2, 300])  # table 3, constantine-2017

        circuit_model = CircuitModel()
        n_samples = M_constantine
        iN_vec = n_samples.astype(int)  # 8 if calc_second_order = False, else 14

        no_runs = np.zeros(len(iN_vec))
        indices = np.zeros(shape=(len(iN_vec), dim))
        indices_error = np.zeros(shape=(len(iN_vec), dim))
        idx = 0
        n_trials = 1
        no_runs_averaged = 1

        dist = cp.J(cp.Uniform(x_lower[0], x_upper[0]), cp.Uniform(x_lower[1], x_upper[1]),
                    cp.Uniform(x_lower[2], x_upper[2]), cp.Uniform(x_lower[3], x_upper[3]),
                    cp.Uniform(x_lower[4], x_upper[4]), cp.Uniform(x_lower[5], x_upper[5]))

        for iN in iN_vec:
            tmp_indices_error_av = np.zeros(dim)
            for i_trial in range(0, n_trials):
                seed = int(np.random.rand(1) * 2 ** 32 - 1)
                random_state = RandomState(seed)

                # https://github.com/jonathf/chaospy/issues/81

                dist_samples = dist.sample(iN)  # random samples or abscissas of polynomials ?
                values_f, _, _ = circuit_model.eval_model_averaged(dist_samples, no_runs_averaged,
                                                                   random_state=random_state)
                # Approximation with Chaospy
                poly = cp.orth_ttr(order_cp, dist)
                approx_model = cp.fit_regression(poly, dist_samples, values_f)
                tmp_indices_total = cp.Sens_t(approx_model, dist)

                tmp_error = relative_error_constantine_2017(tmp_indices_total, sobol_indices_quad_constantine)
                tmp_indices_error_av = tmp_indices_error_av + tmp_error
                print(iN)

            indices_error[idx, :] = tmp_indices_error_av / n_trials
            indices[idx, :] = tmp_indices_total
            no_runs[idx] = iN
            idx = idx + 1

        if bool_plot:
            col = np.array(
                [[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980], [0.9290, 0.6940, 0.1250], [0.4940, 0.1840, 0.5560],
                 [0.4660, 0.6740, 0.1880], [0.3010, 0.7450, 0.9330]])

            plt.figure()
            for i in range(0, dim):
                plt.semilogx(no_runs, indices[:, i], '.--', label='%s (SALib)' % key[i], color=col[i, :])
                plt.semilogx([no_runs[0], max(no_runs)], sobol_indices_quad_constantine[i] * np.ones(2), 'k:',
                             label='Reference values', color=col[i, :])

            plt.xlabel('Number of samples')
            plt.ylabel('Sobol\' total indices')
            plt.legend()

            plt.figure()
            for i in range(0, dim):
                plt.loglog(no_runs, indices_error[:, i], '.--', label=key[i]+'(PC Approximation)', color=col[i, :])
                plt.loglog(M_constantine, sobol_indices_error_constantine[:, i], '.k:', color=col[i, :])

            plt.xlabel('Number of samples')
            plt.ylabel('Relative error (Sobol\' total indices)')
            plt.grid(True, 'minor', 'both')

            plt.legend()
            plt.show()

        # assure that it ran
        assert(True, True)


if __name__ == '__main__':
    unittest.main()
