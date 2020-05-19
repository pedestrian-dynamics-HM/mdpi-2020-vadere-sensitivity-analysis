import unittest
import numpy as np
import matplotlib.pyplot as plt
import numpy.testing as nptest

import chaospy as cp

from uq.utils.model_function import ExponentialModel, CircuitModel
from uq.active_subspace.active_subspace_with_gradients import active_subspace_with_gradients
from uq.active_subspace.active_subspace_with_gradients import calc_eigenvalues, calc_activity_scores_from_C
from uq.active_subspace.utils import transform_coordinates_from_unit, \
    assert_allclose_eigenvectors, relative_error_constantine_2017
from uq.utils.prior_distribution import UniformGenMult


class TestConvergenceActiveSubspace(unittest.TestCase):
    def test_convergence_testmodel(self, bool_plot: bool = False):
        rtol_activity_scores = 1e-1
        testmodel = ExponentialModel()
        density_type = "uniform"
        dim = testmodel.get_dimension()
        x_lower = -1.0 * np.ones(shape=dim)  # lower bounds for parameters
        x_upper = np.ones(shape=dim)  # lower bounds for parameters
        test_input = x_lower
        k = 1 + 1
        bool_gradient = True
        M_boot = 0
        step_size = None
        case = None
        bool_averaged = False
        no_runs_averaged = 1
        bool_save_data = False
        bool_print = False
        path2results = None
        key = ["x1", "x2"]

        alpha_vec = np.array([5e1, 1e2, 5e2, 1e3, 5e3, 1e4]) / np.log(dim) / k
        idx = 0
        activity_scores = np.zeros(shape=(dim, len(alpha_vec)))
        error_c_gradients = np.zeros(shape=len(alpha_vec))
        error_eig = np.zeros(shape=(dim, len(alpha_vec)))
        error_as = np.zeros(shape=(dim, len(alpha_vec)))

        n_samples = np.zeros(len(alpha_vec))
        n_trials = 10
        for alpha in alpha_vec:
            averaged_error_eig = np.zeros(shape=(1, dim))
            averaged_error_as = np.zeros(shape=(1, dim))
            averaged_scores = np.zeros(shape=(1, dim))
            all_scores = np.zeros(shape=(n_trials, dim))
            for j in range(0, n_trials):
                seed = int(np.random.rand(1) * (2 ** 32 - 1))

                # Active Subspaces
                max_rel_error_eig, error_c_gradients[idx], tmp_activity_scores, \
                true_activity_scores, size_subspace, path_to_files, n_samples[idx], lambda_eig, \
                w_active, test_y, lambda_eig_true, idx_gap, idx_gap_true, distance_subspace, true_distance = \
                    active_subspace_with_gradients(
                        testmodel, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                        step_size, step_size, case,
                        seed, bool_averaged, no_runs_averaged, bool_save_data, bool_print, bool_plot, path2results)

                tmp_error_i = np.abs(lambda_eig - lambda_eig_true) / lambda_eig_true
                averaged_error_eig = averaged_error_eig + tmp_error_i
                averaged_scores = averaged_scores + tmp_activity_scores
                all_scores[j, :] = tmp_activity_scores
                tmp_error_as_i = np.abs(tmp_activity_scores - true_activity_scores) / true_activity_scores
                averaged_error_as = averaged_error_as + tmp_error_as_i

                true_activity_scores_save = true_activity_scores
                nptest.assert_almost_equal(true_activity_scores_save, true_activity_scores)


            error_eig[:, idx] = averaged_error_eig / n_trials
            error_as[:, idx] = averaged_error_as / n_trials
            activity_scores[:, idx] = averaged_scores / n_trials
            nptest.assert_almost_equal(np.mean(all_scores, axis=0), activity_scores[:, idx])

            idx = idx + 1

        # Compare the approximated scores to the true scores
        # print(abs_relative_error(activity_scores[:, -1], true_activity_scores))
        nptest.assert_allclose(activity_scores[:, -1], true_activity_scores, rtol=0.1)
        error_min_samples = np.abs(activity_scores[:, 0] - true_activity_scores) / true_activity_scores
        error_max_samples = np.abs(activity_scores[:, -1] - true_activity_scores) / true_activity_scores

        # make sure that the result with most samples is better than the results with the lowest number of samples
        self.assertTrue((error_min_samples >= error_max_samples).all())

        # check the convergence of the results
        for i in range(0, dim):
            relative_error_tmp_i = np.abs(activity_scores[i, :] - true_activity_scores[i])  / true_activity_scores[i]
            relative_error_tmp_i = error_as[i, :]
            p = np.polyfit(np.log10(n_samples), np.log10(relative_error_tmp_i), 1)
            # Accuracy rate of M^(-1/2) - Monte Carlo

            nptest.assert_almost_equal(p[0], -0.5, decimal=1)

            if bool_plot:
                p1 = np.poly1d(p)
                plt.figure()
                plt.loglog(n_samples, relative_error_tmp_i, 'o-', label='Data')
                plt.loglog(n_samples, np.power(10, p1(np.log10(n_samples))), label='Fit: Slope %.2f' % p[0])
                p2 = np.poly1d([-0.5, p[1]])
                plt.loglog(n_samples, np.power(10, p2(np.log10(n_samples))), label='-1/2 Slope')
                print(p1)
                print(p2)
                plt.legend()
                plt.show()



        if bool_plot:
            plt.figure()
            for i in range(0, dim):
                plt.loglog(n_samples, activity_scores[i, :], 'o--', label="Estimated score %s" % key[i])

                plt.loglog(n_samples, true_activity_scores[i] * np.ones(len(alpha_vec)), 'k:',
                           label="True score %s" % key[i])
            plt.xlabel("Number of samples")
            plt.ylabel("Activity score")
            plt.legend()

            plt.figure()
            for i in range(0, dim):
                plt.loglog(n_samples,
                           np.abs(activity_scores[i, :] - true_activity_scores[i] * np.ones(len(alpha_vec))) /
                           true_activity_scores[i], 'o--', label="Estimated activity score %s" % key[i])
                plt.loglog(n_samples, error_eig[i, :], 'x:', label='Estimated eigenvalues of C')

            plt.plot()
            plt.loglog(n_samples, error_c_gradients, label='Estimated C')
            plt.xlabel("Number of samples")
            plt.ylabel("Error")
            plt.legend()

            plt.show()

    def test_calc_eigenvalues(self):
        tmp = np.random.rand(10, 10)
        C = np.matmul(tmp, tmp.transpose())
        # Check eigendecomposition result (distance to C_hat)
        lambda_eig, lambda_eig_true, w_vec, wh = calc_eigenvalues(C, None, None, False)

        # Compare original C with composed C
        np.testing.assert_array_almost_equal(np.matmul(w_vec * lambda_eig, wh), C)

    def test_convergence_circuit_model(self, bool_plot: bool = False):
        x_lower = np.array([50, 25, 0.5, 1.2, 0.25, 50])  # table 3, constantine-2017
        x_upper = np.array([150, 70, 3.0, 2.5, 1.2, 300])  # table 3, constantine-2017

        activity_scores_constantine = np.array(
            [2.377860943309341, 1.619026815363377, 0.261741461441246, 0.075234628507027, 0.000000116801952,
             0.000065807942335])  # calculated with constantine's matlab scripts (constantine-2017)
        first_eigenvector_constantine = np.array(
            [0.740716965496176, -0.611203856808294, -0.245751018607724, 0.131755257933889, 0.000164166339828,
             0.003896711210227])
        eigenvalues_C_hat_constantine = np.array(
            [4.333929773365277, 0.172154546775767, 0.043837280605887, 0.008740767183207, 0.000130619772892,
             0.000000006588527])

        relative_error_alpha_constantine = \
            np.array([[1.0336576094e-01, 7.4484410837e-02, 3.2199103665e-02, 2.3745185339e-02, 7.3669706642e-03,
                       7.6656239406e-03, 3.1840067208e-03],
                      [3.7870122253e-02, 2.9914292211e-02, 1.5144943581e-02, 8.7714103148e-03, 4.4700495080e-03,
                       2.3181577249e-03, 1.1608862367e-03],
                      [5.5327059088e-03, 3.8373262294e-03, 2.1444430111e-03, 1.2336853412e-03, 6.8385574590e-04,
                       4.6215573641e-04, 1.6012529246e-04],
                      [4.7273235250e-03, 3.6211927084e-03, 1.7887246903e-03, 1.0458653419e-03, 6.1191072469e-04,
                       4.0120287646e-04, 1.2149120835e-04],
                      [9.2850207014e-09, 5.7034221317e-09, 3.2097814749e-09, 1.9626299343e-09, 7.6701151806e-10,
                       4.7641920831e-10, 2.2068876806e-10],
                      [8.8701371027e-06, 5.8444088089e-06, 2.9941349453e-06, 1.8812300039e-06, 8.3129987185e-07,
                       5.5281073089e-07, 2.2441932087e-07]])

        relative_error_w_constantine = \
            np.array([[2.7714612059e-02, 1.5914810891e-02, 7.2787857034e-03, 5.6092823993e-03, 2.0922412569e-03,
                       1.5990443476e-03, 5.9951083677e-04],
                      [2.6597648919e-02, 1.8193945825e-02, 8.0722129599e-03, 5.7725628927e-03, 2.3120501167e-03,
                       1.7137227490e-03, 6.8505176689e-04],
                      [1.6522393310e-02, 8.9553701043e-03, 4.2085081539e-03, 3.1165970129e-03, 1.1959665435e-03,
                       7.5160538294e-04, 3.5689233942e-04],
                      [1.5741593060e-02, 1.0683193332e-02, 4.3969836718e-03, 3.4894171678e-03, 1.4456638503e-03,
                       8.6030138247e-04, 3.7841702257e-04],
                      [2.1623799573e-05, 1.1406089340e-05, 8.2572277276e-06, 4.7396837444e-06, 1.9973336116e-06,
                       1.3417606303e-06, 5.2939720012e-07],
                      [8.0536892552e-04, 4.8504474422e-04, 3.4402170980e-04, 1.7722519510e-04, 8.3146852712e-05,
                       5.4381142033e-05, 2.1796009955e-05]])

        circuit_model = CircuitModel()
        dim = circuit_model.get_dimension()
        density_type = "uniform"

        test_input = x_lower
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
        key = ["$R_b1$", "$R_b2$", "$R_f$", "$R_c1$", "$R_c2$", "$beta$"]

        alpha_vec = np.array([5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4]) / np.log(dim) / k

        # Gauss-Legendre quadrature approximation of C
        # abscissas1, weights1 = np.polynomial.legendre.leggauss(np.power(7,6)) -> MemoryError

        distribution = cp.J(cp.Uniform(-1, 1), cp.Uniform(-1, 1), cp.Uniform(-1, 1), cp.Uniform(-1, 1),
                            cp.Uniform(-1, 1), cp.Uniform(-1, 1))
        abscissas, weights = cp.generate_quadrature(dim, distribution, rule="gauss_legendre")
        abscissas_scaled = transform_coordinates_from_unit(x_lower, x_upper, abscissas)

        assert ((np.max(abscissas_scaled, axis=1) <= x_upper).all())
        assert ((np.max(abscissas_scaled, axis=1) >= x_lower).all())

        gradient_evals = circuit_model.eval_gradient(abscissas_scaled)

        # transformation of weights for interval [a,b] instead of [-1,1] -> factor (b-a)/2
        gradient_evals_scaled = np.matmul(0.5 * np.diag(x_upper - x_lower), gradient_evals)

        gradient_evals_scaled_weighted = np.multiply(gradient_evals_scaled, np.sqrt(weights))
        C_tmp = np.matmul(gradient_evals_scaled_weighted, np.transpose(gradient_evals_scaled_weighted))

        n_samples = len(weights)
        tmp_sum = np.zeros(shape=(dim, dim))
        for i in range(0, n_samples):
            tmp = np.expand_dims(gradient_evals_scaled[:, i], axis=1) * gradient_evals_scaled[:, i]
            tmp_sum = tmp_sum + weights[i] * tmp
        C_quadrature_approximation = tmp_sum

        nptest.assert_array_almost_equal(C_quadrature_approximation, C_tmp)

        # calc eigenvalues
        rho = UniformGenMult(x_lower, x_upper, dim)
        activity_scores_quad, __, w_active_quad, __, __, __, __, __, __, lambda_quad, __, __ = calc_activity_scores_from_C(
            C_quadrature_approximation, circuit_model, rho, True, dim, force_idx_gap=0)

        first_eigenvector_quad = w_active_quad[:, 0]

        nptest.assert_array_almost_equal(lambda_quad, eigenvalues_C_hat_constantine)
        nptest.assert_array_almost_equal(activity_scores_quad, activity_scores_constantine)
        assert_allclose_eigenvectors(first_eigenvector_quad, first_eigenvector_constantine)

        ######################################################## Calc activity scores with Algorithm 1.1
        idx = 0
        activity_scores = np.zeros(shape=(dim, len(alpha_vec)))
        activity_score_error_av = np.zeros(shape=(dim, len(alpha_vec)))

        first_eigenvector = np.zeros(shape=(dim, len(alpha_vec)))
        first_eigenvector_error_av = np.zeros(shape=(dim, len(alpha_vec)))
        n_samples = np.zeros(len(alpha_vec))
        n_trials = 10

        for alpha in alpha_vec:
            activity_score_av = np.zeros(shape=dim)
            first_eigenvector_entry_av = np.zeros(shape=dim)
            tmp_activity_score_error_av = np.zeros(shape=dim)
            tmp_first_eigenvector_error_av = np.zeros(shape=dim)
            for trial in range(0, n_trials):  # in constantine-2017 scores are averaged over 10 trials
                seed = int(np.random.rand(1) * (2 ** 32 - 1))
                # Active Subspaces
                max_rel_error_eig, error_c_gradients, tmp_activity_score, \
                __, size_subspace, path_to_files, tmp_n_samples, lambda_eig, \
                w_active, test_y, __, idx_gap, __, distance_subspace, __ = \
                    active_subspace_with_gradients(
                        circuit_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                        step_size, step_size, case, seed, bool_averaged, no_runs_averaged, bool_save_data, bool_print,
                        bool_plot, path2results, force_idx_gap=0)

                # normalize
                nptest.assert_array_almost_equal(np.linalg.norm(w_active[:, 0]), 1)
                tmp_first_eigenvector = w_active[:, 0] * np.sign(w_active[0, 0])

                first_eigenvector_entry_av = first_eigenvector_entry_av + tmp_first_eigenvector
                activity_score_av = activity_score_av + tmp_activity_score
                tmp_activity_score_error = relative_error_constantine_2017(tmp_activity_score, activity_scores_quad)
                tmp_activity_score_error_av = tmp_activity_score_error_av + tmp_activity_score_error

                tmp_first_eigenvector_error = relative_error_constantine_2017(
                    tmp_first_eigenvector * np.sign(tmp_first_eigenvector[0]),
                    first_eigenvector_quad * np.sign(first_eigenvector_quad[0]))
                tmp_first_eigenvector_error_av = tmp_first_eigenvector_error_av + tmp_first_eigenvector_error

            n_samples[idx] = tmp_n_samples
            activity_scores[:, idx] = activity_score_av / n_trials
            activity_score_error_av[:, idx] = tmp_activity_score_error_av / n_trials
            first_eigenvector[:, idx] = first_eigenvector_entry_av / n_trials
            first_eigenvector_error_av[:, idx] = tmp_first_eigenvector_error_av / n_trials

            idx = idx + 1

        # compare this implementation with results of constantine from constantine-2017 (Matlab code)
        nptest.assert_allclose(activity_score_error_av, relative_error_alpha_constantine, rtol=1)
        nptest.assert_allclose(first_eigenvector_error_av, relative_error_w_constantine, rtol=1)

        # check convergence rate
        for i in range(0, dim):
            p = np.polyfit(np.log10(n_samples), np.log10(activity_score_error_av[i, :]), 1)
            # Convergence rate of M^(-1/2) according to constantine-2017
            nptest.assert_almost_equal(p[0], -0.5, decimal=1)

        if bool_plot:
            col = np.array(
                [[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980], [0.9290, 0.6940, 0.1250], [0.4940, 0.1840, 0.5560],
                 [0.4660, 0.6740, 0.1880], [0.3010, 0.7450, 0.9330]])

            # Plot relative error in activity scores
            plt.figure()
            for i in range(0, dim):
                # plt.loglog(n_samples, relative_error_alpha[i, :], 'o-', label=key[i], color=col[i, :])
                plt.loglog(n_samples, activity_score_error_av[i, :], 'o-', label=key[i], color=col[i, :])
                plt.loglog(n_samples, relative_error_alpha_constantine[i, :], 'x:', color=col[i, :])
            plt.ylabel("Relative error (activity score)")
            plt.xlabel("Number of MC samples")
            plt.legend()
            plt.ylim([1e-9, 1e0])
            plt.xlim([5e1, 5e4])
            plt.grid()

            # Plot relative error in first eigenvector scores
            plt.figure()
            for i in range(0, dim):
                plt.loglog(n_samples, first_eigenvector_error_av[i, :], 'o-', label=key[i], color=col[i, :])
                plt.loglog(n_samples, relative_error_w_constantine[i, :], 'x:', color=col[i, :])

            plt.ylabel("Relative error (first eigenvector)")
            plt.xlabel("Number of MC samples")
            plt.legend()
            plt.ylim([1e-9, 1e0])
            plt.xlim([5e1, 5e4])
            plt.grid()
            plt.show()


if __name__ == '__main__':
    unittest.main()
