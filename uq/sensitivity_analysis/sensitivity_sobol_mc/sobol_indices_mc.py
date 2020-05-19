import numpy as np
from numpy.random import RandomState
import warnings
import time

from uq.utils.DataSaver import DataSaver


# m: number of input parameters
# M: number of Monte Carlo points
# from constantine-2017 (Matlab code)
def calc_sobol_indices(model: "Model", rho: "UniformGenMult", m: int, M: int, random_state: RandomState = None,
                       no_runs_averaged: int = 1, path2results: str = None):
    if path2results is not None:
        data_saver = DataSaver(path2results)
        model.set_data_saver(data_saver)
    else:
        data_saver = None
        warnings.warn('No path2results input for DataSaver! ')

    start = time.time()

    N = int(np.floor(M / (1 + m)))
    if N <= 2:  # otherwise the variance is 0, leading to infinity indices
        N = 2
        warnings.warn('Number of samples N resulted in 0 so it was set to 2. ')
    # generate random samples
    samples = rho.sample(int(2 * N), random_state)
    total_indices_constantine, total_indices_jansen, first_indices_jansen, first_incides_saltelli_2010, eval_time = \
        sobol_indices(model, samples, N, m, random_state, no_runs_averaged, bool_first_order=True)

    computation_time = ((time.time() - start) / 60)

    # Save results
    if data_saver is not None:
        data_saver.write_sens_results_to_file(model, rho.get_lower(), rho.get_upper(), no_runs_averaged, M,
                                              total_indices_constantine, total_indices_jansen, first_indices_jansen,
                                              first_incides_saltelli_2010, computation_time)

    return total_indices_constantine, total_indices_jansen, first_indices_jansen, first_incides_saltelli_2010, eval_time


def sobol_indices(model: "Model", samples: np.ndarray, N: int, m: int, random_state: RandomState = None,
                  no_runs_averaged: int = 1, bool_first_order: bool = False):
    # Split up samples in two parts: A samples
    A = samples[:, 0:N]
    start = time.time()
    eval_A, __, __ = model.eval_model_averaged(A, random_state=random_state, no_runs_averaged=no_runs_averaged)
    eval_time = time.time() - start

    variance_A = np.var(eval_A)

    # Split up samples in two parts: B samples
    B = samples[:, range(N, 2 * N)]
    # for first order indices
    if bool_first_order:
        start = time.time()
        eval_B, __, __ = model.eval_model_averaged(B, random_state=random_state, no_runs_averaged=no_runs_averaged)
        eval_time = time.time() - start

        variance_AB = np.var(np.concatenate((eval_A, eval_B), axis=0))

    total_order_constantine = np.zeros(shape=m)

    if bool_first_order:
        first_order_jansen = np.zeros(shape=m)
        first_order_saltelli_2010 = np.zeros(shape=m)
        total_order_jansen = np.zeros(shape=m)
    else:
        first_order_jansen = None
        first_order_saltelli_2010 = None
        total_order_jansen = None

    for i in range(0, m):
        # it is important to make deep copies here!
        ABi = A.copy()
        ABi[i, :] = B.copy()[i, :]
        start = time.time()
        eval_ABi, __, __ = model.eval_model_averaged(ABi, random_state=random_state,
                                                     no_runs_averaged=no_runs_averaged)
        eval_time = eval_time + (time.time() - start)
        tmp_total_jansen = eval_A - eval_ABi
        total_order_constantine[i] = np.dot(tmp_total_jansen, tmp_total_jansen) / (
                2 * N * variance_A)  # code from constantine-2017

        if bool_first_order:
            total_order_jansen[i] = np.dot(tmp_total_jansen, tmp_total_jansen) / (
                    2 * N * variance_AB)  # formula from Jansen-1999 (Saltelli-2010)
            tmp_first_jansen = eval_B - eval_ABi
            first_order_jansen[i] = 1 - (np.dot(tmp_first_jansen, tmp_first_jansen)/(2*N) / variance_AB)  # formula from Jansen-1999 (Saltelli-2010)
            first_order_saltelli_2010[i] = np.dot(eval_B, -tmp_total_jansen) / (N * variance_AB)  # formula from Saltelli-2010

    return total_order_constantine, total_order_jansen, first_order_jansen, first_order_saltelli_2010, eval_time / 60,
