import numpy as np
from numpy.random import RandomState
import time
import matplotlib.pyplot as plt


from uq.utils.prior_distribution import UniformGenMult
from uq.sensitivity_analysis.config import configure_vadere_sa
from uq.sensitivity_analysis.sensitivity_sobol_mc.sobol_indices_mc import calc_sobol_indices
from uq.utils.DataSaver import DataSaver

bool_averaged = True
no_runs_averaged = 10  # average over multiple runs?

M_vec = [10, 50, 100, 500, 1000]
n_runs = 3  # number of runs for each number of samples

run_local = False

scenario_name = "Liddle_osm_v4.scenario"

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

rho = UniformGenMult(x_lower, x_upper, dim)
__, m, path2tutorial = configure_vadere_sa(run_local, scenario_name, key, qoi)


if __name__ == "__main__":  # main required by Windows to run in parallel

    data_saver = DataSaver(path2tutorial, 'summary')

    # ----------------------------------------------------------------  allocation

    seeds_vec = np.round(general_random_state.rand(n_runs) * (2 ** 31)).astype(np.int)

    total_indices_mc = -1 * np.ones(shape=(len(M_vec), n_runs, m))
    first_indices_mc = -1 * np.ones(shape=(len(M_vec), n_runs, m))
    normalized_indices_mc = -1 * np.ones(shape=(len(M_vec), n_runs, m))

    # ----------------------------------------------------------------  perform calculations

    count = 0
    for iM in range(0, len(M_vec)):
        for idx in range(0, n_runs):
            start = time.time()
            seed = seeds_vec[idx]  # use the same seeds for all configs
            # print(seed)

            # configure setup (make sure no old infos are stored)
            vadere_model, m, path2tutorial = configure_vadere_sa(run_local, scenario_name, key, qoi)

            total_indices_constantine, total_indices_jansen, first_indices_jansen, first_incides_saltelli_2010, computation_time_samples = \
                calc_sobol_indices(vadere_model, rho, m=dim, M=M_vec[iM], random_state=RandomState(seed),
                                   no_runs_averaged=no_runs_averaged, path2results=data_saver.get_path_to_files())
            if total_indices_jansen is not None:
                total_indices_mc[iM, idx, :] = total_indices_jansen
                first_indices_mc[iM, idx, :] = first_incides_saltelli_2010
            else:
                total_indices_mc[iM, idx, :] = total_indices_constantine

            count = count + 1

    # --------------------------------------------------------------- Save results

    data_saver.write_var_to_file(total_indices_mc, 'sobol_indices_mc')
    data_saver.write_var_to_file(general_seed, 'general_seed')


    # --------------------------------------------------------------- Evaluation

    print("Finished calculations - Start of evaluation of results")

    h = plt.figure()
    sobol_total_indices_reshaped = np.reshape(total_indices_mc, newshape=(int(total_indices_mc.size / len(key)), -1))

    for i in range(0, n_runs):
        # plt.plot(sobol_indices_mc[i,:, :], 'o:', label=key[i])
        plt.plot(sobol_total_indices_reshaped[i, :], 'o:',
                 label="run %d, alpha = %d" % (i, np.repeat(M_vec, n_runs)[i]))

    plt.xlabel('Parameter index')
    plt.ylabel('Sobol total indices')
    plt.xticks(np.arange(0, dim))
    data_saver.save_figure(h, 'sobol_total_order_indices')

    h = plt.figure()
    sobol_first_indices_reshaped = np.reshape(first_indices_mc, newshape=(int(first_indices_mc.size / len(key)), -1))

    for i in range(0, n_runs):
        # plt.plot(sobol_indices_mc[i,:, :], 'o:', label=key[i])
        plt.plot(sobol_first_indices_reshaped[i, :], 'o:',
                 label="run %d, alpha = %d" % (i, np.repeat(M_vec, n_runs)[i]))

    plt.xlabel('Parameter index')
    plt.ylabel('First order Sobol indices')
    plt.xticks(np.arange(0, dim))
    data_saver.save_figure(h, 'sobol_first_order_indices')

    # h = plt.figure()
    # for i in range(0, n_runs):
    #    plt.plot(normalized_indices_mc[i,:, :], 'o:', label=key[i])#

    # plt.xlabel('Parameter index')
    # plt.ylabel('Normalized Sobol indices')
    # plt.xticks(np.arange(0, dim))
    # data_saver.save_figure(h, 'normalized_sobol_indices')

    # plt.show()
