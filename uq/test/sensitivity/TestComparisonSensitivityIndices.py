import unittest
import numpy.testing as nptest
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState

from uq.utils.model_function import ExponentialModel, IshigamiFunction
from uq.active_subspace.active_subspace_with_gradients import active_subspace_with_gradients
from uq.active_subspace.config_matrix_constantine_fct import config_matrix_constantine
from uq.sensitivity_analysis.sensitivity_sobol_mc.sobol_indices_mc import calc_sobol_indices
from uq.utils.prior_distribution import UniformGenMult


# Compares the indices obtained with Active Subspaces and with Sobol' method
class TestSensitivityIndicesTestModel(unittest.TestCase):

    def test_compare_indices_quadratic_model(self, bool_plot: bool = False):
        case = 2
        seed = 354365
        print("gradients_compared_to_constantine_one_case:  CASE " + str(case))
        bool_gradient = True
        step_size = None
        alpha = 20
        k = 10
        M_boot = 0
        M = 5000
        N = 1000

        bool_save_data = False
        bool_print = False
        bool_plot = False

        path2results = None
        bool_averaged = None
        no_runs_averaged = 1

        ## Model
        __, x_lower, x_upper, m, density_type, test_input = config_matrix_constantine(case, RandomState(seed))

        ## Activity scores with true gradients
        quadratic_model, __, __, __, __, __ = config_matrix_constantine(case, RandomState(seed))  # re-init
        rho = UniformGenMult(x_lower, x_upper, m)
        max_rel_error_eig, error_c_gradients, activity_scores, true_activity_scores, size_subspace, path_to_files, \
        n_samples, lambda_eig, w_active, test_y, lambda_eig_true, idx_gap, idx_gap_true, distance_subspace, \
        true_distance = \
            active_subspace_with_gradients(
                quadratic_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size, step_size, case, seed, bool_averaged, no_runs_averaged, bool_save_data, bool_print,
                bool_plot, path2results)

        ## Sobol' indices with MC method
        quadratic_model, __, __, __, __, __ = config_matrix_constantine(case, RandomState(seed))  # re-init
        rho = UniformGenMult(x_lower, x_upper, m)
        total_indices_constantine, total_indices_jansen, first_indices_jansen, first_incides_saltelli_2010, computation_time_samples = \
            calc_sobol_indices(quadratic_model, rho, m=m, M=M, random_state=RandomState(seed),
                               no_runs_averaged=no_runs_averaged, path2results=None)

        ## Sobol indices with SALib
        quadratic_model, __, __, __, __, __ = config_matrix_constantine(case, RandomState(seed))  # re-init
        # Defining the Model Inputs
        key = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]
        problem = {
            'num_vars': len(key),
            'names': key,
            'bounds': np.column_stack((x_lower, x_upper)).tolist()
        }

        nptest.assert_almost_equal(np.abs(activity_scores) / np.sqrt(np.sum(np.square(activity_scores))),
                                   np.abs(total_indices_jansen) / np.sqrt(np.sum(np.square(total_indices_jansen))),
                                   decimal=1)

        if bool_plot:
            ## Plot resulting scores

            # Normalization according constantine-2017 -> necessary since activity scores are extremely large!
            plt.figure()
            plt.plot(np.abs(activity_scores) / np.sqrt(np.sum(np.square(activity_scores))), 'o:',
                     label="Activity scores")
            plt.plot(np.abs(total_indices_jansen) / np.sqrt(np.sum(np.square(total_indices_jansen))), 's-.',
                     label='Sobol\' total effect indices (Jansen)')
            plt.legend()
            plt.show()

    def test_compare_indices_testmodel(self, bool_plot: bool = False):
        rtol_scores = 1

        testmodel = ExponentialModel()
        density_type = "uniform"
        dim = testmodel.get_dimension()
        x_lower = -1.0 * np.ones(shape=dim)  # lower bounds for parameters
        x_upper = np.ones(shape=dim)  # lower bounds for parameters
        rho = UniformGenMult(x_lower, x_upper, dim)

        test_input = x_lower
        alpha = 10
        k = 1 + 1
        M = 100
        N = 500

        bool_gradient = True
        M_boot = 0
        step_size = None
        case = None
        seed = 2456354
        bool_averaged = False
        no_runs_averaged = 1
        bool_save_data = False
        bool_print = False
        bool_plot = False
        path2results = None

        # Active Subspaces
        max_rel_error_eig, error_c_gradients, activity_scores, \
        true_activity_scores, size_subspace, path_to_files, n_samples, lambda_eig, \
        w_active, test_y, lambda_eig_true, idx_gap, idx_gap_true, distance_subspace, true_distance = \
            active_subspace_with_gradients(
                testmodel, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                step_size, step_size, case,
                seed, bool_averaged, no_runs_averaged, bool_save_data, bool_print, bool_plot, path2results)

        # Sobol indices (SA Lib)
        seed = 78613651
        no_runs_averaged = 1

        # Defining the Model Inputs
        key = ["x1", "x2"]
        problem = {
            'num_vars': len(key),
            'names': key,
            'bounds': np.column_stack((x_lower, x_upper)).tolist()
        }


        total_indices_constantine, total_indices_jansen, first_indices_jansen, first_incides_saltelli_2010, computation_time_samples = \
            calc_sobol_indices(testmodel, rho, m=dim, M=M, random_state=RandomState(seed),
                               no_runs_averaged=no_runs_averaged, path2results=None)


        # Compare scores by Jansen / Saltelli with activity scores for test model
        nptest.assert_allclose(total_indices_jansen, activity_scores, rtol=0.1)

        if bool_plot:
            plt.figure()
            plt.plot(activity_scores, 'o:', label="Activity scores")
            plt.plot(total_indices_jansen, 's-.', label='Sobol\' total effect indices (Jansen method)')
            plt.legend()
            plt.show()

    def test_ishigami_indices(self, bool_plot: bool = False):
        # Parameters according to Sobol' and levitan (1999)
        model = IshigamiFunction(a=7, b=0.05)
        dim = model.get_dimension()
        M_vec = np.array([50, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5]).astype(int)
        N_vec = 2 * (np.floor(M_vec / (1 + model.get_dimension()))).astype(int)

        x_lower = np.array([-np.pi, -np.pi, -np.pi])  # https://uqworld.org/t/ishigami-function/55
        x_upper = np.array([np.pi, np.pi, np.pi])  # https://uqworld.org/t/ishigami-function/55

        rho = UniformGenMult(x_lower, x_upper, dim)

        n_trials = 50
        total_indices_constantine = np.zeros(shape=(len(M_vec), n_trials, dim))
        total_indices_jansen = np.zeros(shape=(len(M_vec), n_trials, dim))
        first_indices_jansen = np.zeros(shape=(len(M_vec), n_trials, dim))
        first_incides_saltelli_2010 = np.zeros(shape=(len(M_vec), n_trials, dim))

        err_total_constantine = np.zeros(shape=(len(M_vec), n_trials, dim))
        err_total_jansen = np.zeros(shape=(len(M_vec), n_trials, dim))
        err_first_jansen = np.zeros(shape=(len(M_vec), n_trials, dim))
        err_first_saltelli_2010 = np.zeros(shape=(len(M_vec), n_trials, dim))

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

                err_total_constantine[idx, i, :] = np.abs(total_indices_constantine_tmp - true_total_effects)
                err_total_jansen[idx, i, :] = np.abs(total_indices_jansen_tmp - true_total_effects)
                err_first_jansen[idx, i, :] = np.abs(first_indices_jansen_tmp - true_total_effects)
                err_first_saltelli_2010[idx, i, :] = np.abs(first_incides_saltelli_2010_tmp - true_total_effects)

            idx = idx + 1

        ### Compare to exact indices
        nptest.assert_allclose(np.mean(first_incides_saltelli_2010[-1, :, :], axis=0), true_first_order_indices,
                               atol=1e-2)
        nptest.assert_allclose(np.mean(first_indices_jansen[-1, :, :], axis=0), true_first_order_indices, atol=1e-2)

        nptest.assert_allclose(np.mean(total_indices_jansen[-1, :, :], axis=0), true_total_effects, atol=1e-2)
        nptest.assert_allclose(np.mean(total_indices_constantine[-1, :, :], axis=0), true_total_effects, atol=1e-2)

        if bool_plot:
            ### Plot results
            col = np.array(
                [[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980], [0.9290, 0.6940, 0.1250], [0.4940, 0.1840, 0.5560],
                 [0.4660, 0.6740, 0.1880], [0.3010, 0.7450, 0.9330]])
            plt.figure()
            for idim in range(0, dim):
                plt.semilogx(N_vec, true_total_effects[idim] * np.ones(len(M_vec)), '-', color=col[idim, :],
                             label='Exact index (' + model.get_key()[idim] + ')')
                plt.semilogx(N_vec, np.mean(total_indices_jansen[:, :, idim], axis=1), 'x:', color=col[idim, :],
                             label='Jansen-1999 (' + model.get_key()[idim] + ')')
                plt.semilogx(N_vec, np.mean(total_indices_constantine[:, :, idim], axis=1), '+--', color=col[idim, :],
                             label='Saltelli-2010 (' + model.get_key()[idim] + ')')
            plt.legend()
            plt.xlabel('Number of samples (2N)')
            plt.ylabel('Sobol\' total order effects')
            plt.title('Averaged over %d runs' % n_trials)
            plt.show()

            ## error plot (total indices)
            plt.figure()
            for idim in range(0, dim):
                plt.loglog(M_vec, np.mean(
                    np.abs(total_indices_jansen[:, :, idim] - true_total_effects[idim]) / true_total_effects[idim],
                    axis=1),
                           'x:', color=col[idim, :], label='Jansen-1999 (' + model.get_key()[idim] + ')')
                plt.loglog(M_vec,
                           np.mean(np.abs(total_indices_constantine[:, :, idim] - true_total_effects[idim]) /
                                   true_total_effects[idim], axis=1),
                           '+--', color=col[idim, :], label='Saltelli-2010 (' + model.get_key()[idim] + ')')

            y_start = np.mean(np.abs(total_indices_jansen[0, :, :] - true_total_effects) / true_total_effects)
            # plt.xlim([1e1, 1e5])
            x_limits = np.array([np.min(M_vec), np.max(M_vec)])
            slope = -1 / 2
            diff_x = np.log10(x_limits[1]) - np.log10(x_limits[0])
            np.log10(y_start)
            plt.loglog(x_limits, np.array([y_start, np.power(10, np.log10(y_start) + diff_x * slope)]), '-k',
                       label='slope = -1/2')
            plt.legend()
            plt.xlabel('Sampling factor $M$')
            plt.ylabel('Relative error in Sobol\' total order effects')
            plt.title('Averaged over %d runs' % n_trials)
            plt.show()

            ## first order plot
            plt.figure()
            for idim in range(0, dim):
                plt.semilogx(N_vec, true_first_order_indices[idim] * np.ones(len(M_vec)), '-', color=col[idim, :],
                             label='True indices ' + model.get_key()[idim])
                plt.semilogx(N_vec, np.mean(first_incides_saltelli_2010[:, :, idim], axis=1), 'x:',
                             color=col[idim, :])
                plt.semilogx(N_vec, np.mean(first_indices_jansen[:, :, idim], axis=1), '+--', color=col[idim, :])
            plt.legend()
            plt.xlabel('Number of samples (2N)')
            plt.ylabel('Sobol\' first order indices')
            plt.title('Averaged over %d runs' % n_trials)
            plt.show()

            ## error plot (first indices)
            plt.figure()
            for idim in range(0, dim):
                if np.abs(true_first_order_indices[idim]) > 0:
                    plt.loglog(M_vec, np.mean(
                        np.abs(first_indices_jansen[:, :, idim] - true_first_order_indices[idim]) /
                        true_first_order_indices[idim], axis=1),
                               'x:', color=col[idim, :], label='Jansen-1999 (' + model.get_key()[idim] + ')')
                    plt.loglog(M_vec,
                               np.mean(
                                   np.abs(first_incides_saltelli_2010[:, :, idim] - true_first_order_indices[idim]) /
                                   true_first_order_indices[idim], axis=1),
                               '+--', color=col[idim, :], label='Saltelli-2010 (' + model.get_key()[idim] + ')')
                else:
                    plt.loglog(M_vec, np.mean(
                        np.abs(first_indices_jansen[:, :, idim] - true_first_order_indices[idim]), axis=1),
                               'x:', color=col[idim, :], label='Jansen-1999 (' + model.get_key()[idim] + '), absolute')
                    plt.loglog(M_vec,
                               np.mean(np.abs(first_incides_saltelli_2010[:, :, idim] - true_first_order_indices[idim]),
                                       axis=1),
                               '+--', color=col[idim, :],
                               label='Saltelli-2010 (' + model.get_key()[idim] + '), absolute')

            y_start = np.mean(np.abs(total_indices_jansen[0, :, :] - true_total_effects) / true_total_effects)
            # plt.xlim([1e1, 1e5])
            x_limits = np.array([np.min(M_vec), np.max(M_vec)])
            slope = -1 / 2
            diff_x = np.log10(x_limits[1]) - np.log10(x_limits[0])
            np.log10(y_start)
            plt.loglog(x_limits, np.array([y_start, np.power(10, np.log10(y_start) + diff_x * slope)]), '-k',
                       label='slope = -1/2')
            plt.legend()
            plt.xlabel('Sampling factor $M$')
            plt.ylabel('Relative error in Sobol\' first order effects')
            plt.title('Averaged over %d runs' % n_trials)
            plt.show()


if __name__ == '__main__':
    unittest.main()
