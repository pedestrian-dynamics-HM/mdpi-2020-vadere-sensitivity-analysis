import unittest
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState
import numpy.testing as nptest

from uq.utils.prior_distribution import UniformGenMult
from uq.utils.model_function import CircuitModel, IshigamiFunction
from uq.active_subspace.utils import relative_error_constantine_2017

from uq.sensitivity_analysis.sensitivity_sobol_mc.sobol_indices_mc import calc_sobol_indices


class TestSobolMC(unittest.TestCase):



    def test_ishigami_sobol_indices(self, bool_plot: bool = False):
        # Parameters according to Sobol' and levitan (1999)
        model = IshigamiFunction(a=7, b=0.05)
        dim = model.get_dimension()
        M_vec = np.array([50, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5]).astype(int)

        x_lower = np.array([-np.pi, -np.pi, -np.pi])  # https://uqworld.org/t/ishigami-function/55
        x_upper = np.array([np.pi, np.pi, np.pi])  # https://uqworld.org/t/ishigami-function/55

        rho = UniformGenMult(x_lower, x_upper, dim)

        n_trials = 50
        total_indices_constantine = np.zeros(shape=(len(M_vec), n_trials, dim))
        total_indices_jansen = np.zeros(shape=(len(M_vec), n_trials, dim))
        first_indices_jansen = np.zeros(shape=(len(M_vec), n_trials, dim))
        first_incides_saltelli_2010 = np.zeros(shape=(len(M_vec), n_trials, dim))

        true_first_order_indices = model.get_true_first_order_indices()
        true_total_effects = model.get_true_total_effect_indices()

        idx = 0
        for M in M_vec:
            for i in range(0, n_trials):
                seed = int(np.random.rand(1) * (2 ** 32 - 1))
                total_indices_constantine_tmp, total_indices_jansen_tmp, first_indices_jansen_tmp, first_incides_saltelli_2010_tmp, computation_time_samples = \
                    calc_sobol_indices(model, rho, m=dim, M=M, random_state=RandomState(seed))

                total_indices_constantine[idx, i, :] = total_indices_constantine_tmp
                total_indices_jansen[idx, i, :] = total_indices_jansen_tmp
                first_indices_jansen[idx, i, :] = first_indices_jansen_tmp
                first_incides_saltelli_2010[idx, i, :] = first_incides_saltelli_2010_tmp

            idx = idx + 1

        # Compare to exact indices
        nptest.assert_allclose(np.mean(first_incides_saltelli_2010[-1, :, :], axis=0), true_first_order_indices,
                               atol=1e-2)
        nptest.assert_allclose(np.mean(first_indices_jansen[-1, :, :], axis=0), true_first_order_indices, atol=1e-2)

        nptest.assert_allclose(np.mean(total_indices_jansen[-1, :, :], axis=0), true_total_effects, atol=1e-2)
        nptest.assert_allclose(np.mean(total_indices_constantine[-1, :, :], axis=0), true_total_effects, atol=1e-2)


    def test_sobol_circuit_model(self, bool_plot: bool = False):
        model = CircuitModel()
        dim = model.get_dimension()
        x_lower = np.array([50, 25, 0.5, 1.2, 0.25, 50])  # table 3, constantine-2017
        x_upper = np.array([150, 70, 3.0, 2.5, 1.2, 300])  # table 3, constantine-2017

        rho = UniformGenMult(x_lower, x_upper, dim)

        M_vec = np.array([50, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4]).astype(int)

        key = model.get_key()
        n_trials = 50

        constantine_sobol_indices = np.array(
            [5.0014515064e-01, 4.1167859899e-01, 7.4006053045e-02, 2.1802568214e-02, 5.1736552010e-08,
             1.4938996627e-05])
        constantine_error_indices = \
            np.transpose(
                np.array(
                    [[0.611146228697800, 0.270365434745662, 0.154666380085916, 0.128123675773586, 0.050229955234354,
                      0.035420048253200, 0.014486328386486],
                     [0.600744044901855, 0.320240964568749, 0.122964263656201, 0.096725945245589, 0.053143328175264,
                      0.032748864015757, 0.011486316471843],
                     [0.117896942282141, 0.046150927238921, 0.026268692965229, 0.018450563871356, 0.008365659231769,
                      0.005855097430867, 0.002820892192452],
                     [0.038013619285947, 0.016186288111641, 0.008989392030354, 0.006391124957791, 0.002621904942281,
                      0.001921507769812, 0.000953902244786],
                     [0.000000123407464, 0.000000048204289, 0.000000030780845, 0.000000025240466, 0.000000010551377,
                      0.000000006950614, 0.000000003337215],
                     [0.000034241277775, 0.000018074628532, 0.000007155465971, 0.000005030346761, 0.000002759331399,
                      0.000001952947040, 0.000000728400437]]))

        col = np.array(
            [[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980], [0.9290, 0.6940, 0.1250], [0.4940, 0.1840, 0.5560],
             [0.4660, 0.6740, 0.1880], [0.3010, 0.7450, 0.9330]])

        # ---------------------------------------------------  Calculate indices

        total_indices = np.zeros(shape=(len(M_vec), dim))
        relative_error = np.zeros(shape=(len(M_vec), dim))
        idx = 0
        for M in M_vec:
            indices_average = np.zeros(shape=dim)
            error_average = np.zeros(shape=dim)

            for itrial in range(0, n_trials):
                seed = int(np.random.rand(1) * (2 ** 32 - 1))
                tmp_total_constantine, __, __, __, __ = calc_sobol_indices(model, rho, m=dim, M=M,
                                                                           random_state=RandomState(seed))
                tmp_error = relative_error_constantine_2017(tmp_total_constantine, constantine_sobol_indices)
                error_average = error_average + tmp_error
                indices_average = indices_average + tmp_total_constantine

            total_indices[idx, :] = indices_average / n_trials
            relative_error[idx, :] = error_average / n_trials
            idx = idx + 1

        # compare error in activity scores
        nptest.assert_allclose(relative_error, constantine_error_indices, rtol=10)
        nptest.assert_allclose(total_indices[-1, :], constantine_sobol_indices, rtol=0.1)

        if bool_plot:
            # ---------------------------------------------------  Evaluation
            plt.figure()
            for idim in range(0, dim):
                plt.semilogx(M_vec, constantine_sobol_indices[idim] * np.ones(len(M_vec)), ':', color=col[idim, :])
                plt.semilogx(M_vec, total_indices[:, idim], 'x-', label=key[idim], color=col[idim, :])
            plt.legend()
            plt.xlabel('Number of MC samples')
            plt.ylabel('Total sensitivity indices (Sobol)')

            plt.figure()
            for idim in range(0, dim):
                plt.loglog(M_vec, constantine_error_indices[:, idim], '.:', color=col[idim, :])
                plt.loglog(M_vec, relative_error[:, idim], 'x-', label=key[idim], color=col[idim, :])
            plt.legend()
            plt.xlabel('Number of MC samples')
            plt.ylabel('Relative error (Sobol total sensitivity indices)')
            plt.grid()
            plt.show()


if __name__ == '__main__':
    unittest.main()
