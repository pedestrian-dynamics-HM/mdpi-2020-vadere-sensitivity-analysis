import numpy as np
from numpy.random import RandomState
from uq.active_subspace.active_subspace_with_gradients import active_subspace_with_gradients

from uq.sensitivity_analysis.config import configure_vadere_sa
from uq.utils.DataSaver import DataSaver

bool_gradient = False  # is an analytical gradient available? -> True
bool_averaged = True
no_runs_averaged = 10  # average over multiple runs?

step_size = 0.025  # Number of pedestrians wird um 1 erhöht

alpha_vec = [1, 2, 5, 10, 15, 20]  # oversampling factor
k = 2 + 1  # desired dimension of subspace +1
M_boot = 0  # Constantine: Typically between 100 and 10^5
case = " "
n_runs = 3  # number of runs for each number of samples

run_local = False

scenario_name = "Liddle_osm_v4.scenario"

bool_save_data = True
bool_print = True
bool_plot = True

density_type = "uniform"  # input parameter density

test_input = np.array([[1], [1.34], [0.26], [180], [1.6], [0.5]])  # legal test input

# parameter limits
x_lower = np.array([1.0, 0.5, 0.1, 160, 1.6, 30])  # lower bounds for parameters
x_upper = np.array([5.0, 2.2, 1.0, 200, 3.0, 70])  # upper bounds for parameters

# parameters
key = ["queueWidthLoading", "attributesPedestrian.speedDistributionMean",
       "attributesPedestrian.speedDistributionStandardDeviation",
       "sources.[id==3].spawnNumber", "bottleneck_width",
       "pedPotentialHeight"]  # uncertain parameters

key_str_plot = ["control parameter", "free-flow mean", "free-flow dev", "spawn number", "bottleneck width",
                "obstacle repulsion"]

qoi = "max_density.txt"  # quantity of interest

dim = len(key)

general_seed = 267396767
general_random_state = RandomState(seed=general_seed)  # initialize Random State with the current time

# configure setup
__, m, path2tutorial = configure_vadere_sa(run_local, scenario_name, key, qoi)

if __name__ == "__main__":  # main required by Windows to run in parallel

    # %% perform calculations
    n = len(alpha_vec)

    test_input = x_lower

    # %% allocation
    count = 0
    mean_error_eig_gradients = np.zeros(n)
    mean_error_eig_finite_diff = np.zeros(n)
    mean_error_eig_local = np.zeros(n)

    seeds_vec = np.round(general_random_state.rand(n_runs) * (2 ** 31)).astype(np.int)

    result_finite_diff = -1 * np.ones(shape=(n, n_runs, m))
    result_gradients = -1 * np.ones(shape=(n, n_runs, m))
    error_c_gradients = -1 * np.ones(shape=(n, n_runs))
    activity_scores = -1 * np.ones(shape=(n, n_runs, m))
    true_activity_scores = -1 * np.ones(shape=(n, n_runs, m))
    size_subspace = np.zeros(shape=(n, n_runs))

    path2files = []

    datasaver = DataSaver(path2tutorial, 'summary')

    for idx in range(0, n):
        alpha = alpha_vec[idx]

        tmp_gradients = 0
        tmp_finite_diff = 0
        tmp_local_linear = 0

        for irun in range(0, n_runs):  # average over several runs
            seed = seeds_vec[irun]  # use the same seeds for all configs
            # print(seed)

            # configure setup (make sure no old infos are stored)
            test_model, __, __ = configure_vadere_sa(run_local, scenario_name, key, qoi)

            step_size_relative = step_size * (x_upper - x_lower)

            # True gradients
            result_gradients[idx, irun, :], error_c_gradients[idx, irun], activity_scores[idx, irun, :], \
            true_activity_scores[idx, irun, :], size_subspace[idx, irun], tmp_path2files, n_samples, lambda_eig, \
            w_active, __, __, __, __, __, __ = \
                active_subspace_with_gradients(
                    test_model, density_type, x_lower, x_upper, test_input, alpha, k, bool_gradient, M_boot,
                    step_size_relative, step_size, case, seed, bool_averaged, no_runs_averaged, bool_save_data,
                    bool_print, bool_plot, datasaver.get_path_to_files())

            path2files.append(tmp_path2files)
            tmp_gradients = tmp_gradients + result_gradients[idx, irun]

        count = count + 1

    print("Finished calculations - Start of evaluation of results")

    # Write all results to file
    datasaver.write_var_to_file(general_seed, 'general_seed')
    datasaver.write_var_to_file(path2files, "all_path2files")
    datasaver.write_var_to_file(np.reshape(activity_scores, newshape=(int(activity_scores.size / len(key)), -1)),
                                "all_activity_scores")
    datasaver.write_var_to_file(size_subspace.flatten(), "all_size_subspace")
