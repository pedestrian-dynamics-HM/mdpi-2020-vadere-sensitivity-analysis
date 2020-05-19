import warnings
import numpy as np
import os
import time

from uq.utils.DataSaver import DataSaver
from uq.utils.model_function import ApproxModel, Model
from uq.utils.prior_distribution import UniformGenMult
from uq.active_subspace.utils import divide_active_inactive_subspace, calc_activity_scores, find_largest_gap_log, \
    eigendecomposition, transform_coordinates_to_unit, transform_coordinates_from_unit, distance_subspace_eq_349, \
    distance_subspace_corollary310, bootstrapping, compute_C_from_gradients, relative_error


def calc_activity_scores_from_C(C_hat, test_model: Model, rho, bool_print: bool, dim: int, force_idx_gap: int = None, step_size: float = None):
    lambda_eig, true_lambda_eig, w_vec, __ = calc_eigenvalues(C_hat, test_model, rho, bool_print)

    # find largest gap of eigenvalues
    if force_idx_gap is None:  # standard case
        idx_gap, _ = find_largest_gap_log(lambda_eig, step_size)
    else:  # if the idx gap is given (for examples only)
        warnings.warn(
            'calc_activity_scores_from_C: idx_gap is forced to a certain idx! - Should only be done in tests.')
        idx_gap = force_idx_gap

    lambda_eig_true = None
    if test_model.get_analytic_solution_available():
        lambda_eig_true = test_model.get_eigenvalues_C(rho)
        idx_gap_true, _ = find_largest_gap_log(lambda_eig_true)
        if bool_print:
            if idx_gap_true == idx_gap:
                print('True eigenvalue gap found: OK ')
            else:
                print('True eigenvalue gap not found: FAILED ')
    else:
        idx_gap_true = None

    # divide into active and inactive subspaces
    n = idx_gap + 1
    w_active, w_inactive = divide_active_inactive_subspace(w_vec, idx_gap + 1, dim)

    activity_scores = calc_activity_scores(lambda_eig, w_vec, dim, n)

    dist_as, dist_as_vec, true_distance, true_eig, true_W = \
        calc_distance_active_subspace(test_model, dim, w_vec, rho, lambda_eig, idx_gap, bool_print, n)

    true_activity_scores = None
    if test_model.analytic_solution_available:
        true_activity_scores = calc_activity_scores(true_eig, true_W, dim, n)
        # if bool_print:
        #    print("first eigenvector of true W")
        #    print(true_W[:, 0])

    return activity_scores, true_activity_scores, w_active, w_vec, dist_as, dist_as_vec, true_distance, idx_gap, \
           idx_gap_true, lambda_eig, lambda_eig_true, n


def calc_distance_active_subspace(test_model: Model, dim: int, w_vec, rho, lambda_eig, idx_gap: int, bool_print: bool,
                                  n: int):
    #  true distance
    true_eig = None
    true_W = None
    if test_model.analytic_solution_available:
        true_distance = np.zeros(dim - 1)
        dist_as_vec = np.zeros(dim - 1)
        for subspace_idx in range(0, dim - 1):
            # subspace estimated with M samples
            _, w_inactive_tmp = divide_active_inactive_subspace(w_vec, subspace_idx + 1, dim)

            # true C matrix
            true_C = test_model.get_C_matrix(rho)
            true_W, true_eig, true_Wh = eigendecomposition(true_C)

            # true k-dimensional subspace
            true_W_active_k, _ = divide_active_inactive_subspace(true_W, subspace_idx + 1, dim)

            # distance between true k-dimensional subspace and subspace estimated with M samples
            true_distance[subspace_idx] = distance_subspace_eq_349(true_W_active_k, w_inactive_tmp)

            if subspace_idx < dim:
                dist_as_vec[subspace_idx] = distance_subspace_corollary310(lambda_eig, subspace_idx, n_dim=dim)
    else:
        true_distance = None
        dist_as_vec = None

    dist_as = distance_subspace_corollary310(lambda_eig, idx_gap, n_dim=dim)

    if bool_print:
        print("** quality of active subspace (corollary 3.10)")
        print(dist_as)  # bounded by 1? (constantine-2014, p. 37)
        print("** approx **")

    return dist_as, dist_as_vec, true_distance, true_eig, true_W


# check results of bootstrapping
def calc_distance_subspace(distance_subspace, true_distance, bool_print: bool):
    max_distance = None
    if distance_subspace is not None:
        # error in subspace distance
        max_distance = np.max(np.abs(distance_subspace - true_distance))
        if bool_print:
            print("** Max error in subspace distance")
            print(max_distance)
    return max_distance


def check_C_hat(C_hat, test_model: Model, bool_print: bool, rho):
    error_c_hat = None
    if test_model.analytic_solution_available:
        error_c_hat = np.linalg.norm(C_hat - test_model.get_C_matrix(rho))
        if bool_print:
            print("Error in C_hat:")
            print(error_c_hat)
    return error_c_hat


def calc_eigenvalues(C_hat, test_model: Model, rho: "Prior", bool_print: bool):
    # Calculate eigenvectors and eigenvalues
    w_vec, lambda_eig, wh = eigendecomposition(C_hat)

    if test_model is not None and test_model.get_analytic_solution_available():
        lambda_eig_true = test_model.get_eigenvalues_C(rho)
        if bool_print:
            print(np.linalg.norm(lambda_eig - lambda_eig_true))
    else:
        lambda_eig_true = None

    return lambda_eig, lambda_eig_true, w_vec, wh


#  Algorithm 1.1 from constantine-2015 (p. 4)
def active_subspace_with_gradients(test_model: Model, density_type: str, x_lower, x_upper, test_input, alpha: int,
                                   k: int, bool_gradient: bool, M_boot: int, step_size_relative, step_size, case: int,
                                   seed: int, bool_averaged: bool, no_runs_averaged: int, bool_save_data: bool,
                                   bool_print: bool, bool_plot: bool, path2results, force_idx_gap: int = None):
    start = time.time()

    random_state = np.random.RandomState(seed)

    if not bool_averaged:
        no_runs_averaged = 1

    if bool_print:
        print("Random seed: %d" % seed)
    if path2results is not None:
        data_saver = DataSaver(path2results)
        test_model.set_data_saver(data_saver)
    else:
        data_saver = None

    # Active subspace estimation with gradients
    dim = test_model.get_dimension()

    # density of uncertain parameters
    if density_type == 'uniform':
        rho = UniformGenMult(lower=transform_coordinates_to_unit(x_lower, x_upper, value=x_lower),
                             upper=transform_coordinates_to_unit(x_lower, x_upper, value=x_upper), dim=dim)
    else:
        raise Warning("Density type not yet implemented")

    # test of transformation
    test_x = transform_coordinates_to_unit(x_lower, x_upper, test_input)
    test_y = transform_coordinates_from_unit(x_lower, x_upper, test_x)

    if bool_print:
        print("Diff after transformation (Error)")
        print(np.max(test_y - test_input))

    # generate samples according to density of uncertain parameters
    n_samples = np.ceil(alpha * k * np.log(dim)).astype(np.int)  # number of samples
    if bool_print:
        print("Number of samples to be simulated: %d" % n_samples)

    # samples = np.random.rand(m,M)*2-1
    samples = rho.sample(n_samples, random_state=random_state)

    samples_transformed = transform_coordinates_from_unit(x_lower=x_lower, x_upper=x_upper, value=samples)

    if bool_gradient:
        # evaluate gradient at samples todo: parallelize
        values_grad_f_original = test_model.eval_gradient(samples_transformed)
        # scaling of gradient (according to quadrature rule factor - compensation of coordinate transformation
        values_grad_f = np.matmul(0.5 * np.diag(x_upper - x_lower), values_grad_f_original)
    else:
        # evaluate model at samples todo: parallelize
        # values_f, _, _ = test_model.eval_model_averaged(samples_transformed, no_runs_averaged, random_state=random_state)

        # approximate gradients by finite differences
        values_grad_f_original = test_model.approximate_gradient(input_vector=samples_transformed,
                                                                 step_size=step_size_relative,
                                                                 model_eval_vec=None, n_runs_av=no_runs_averaged,
                                                                 random_state=random_state)

        values_grad_f = np.matmul(0.5 * np.diag(x_upper - x_lower), values_grad_f_original)

        if test_model.analytic_solution_available:

            values_grad_f_real = test_model.eval_gradient(samples_transformed)
            if bool_print:
                print("Max diff between gradient and approximation")
                print(np.max(np.abs(values_grad_f - values_grad_f_real)))

    # %% Compute ^C
    C_hat = compute_C_from_gradients(values_grad_f, n_samples, dim)
    error_c_hat = check_C_hat(C_hat, test_model, bool_print, rho)

    activity_scores, true_activity_scores, w_active, w_vec, dist_as, dist_as_vec, true_distance, idx_gap, \
    idx_gap_true, lambda_eig, lambda_eig_true, n = \
        calc_activity_scores_from_C(C_hat, test_model, rho, bool_print, dim, force_idx_gap)

    # Compare activity scores like constantine
    __, Sigma, Wh = np.linalg.svd(np.transpose(values_grad_f), full_matrices=False)
    W = np.transpose(Wh)
    lambda_constantine = np.square(Sigma) / n_samples
    activity_scores_constantine = np.square(W[:, 0:idx_gap + 1]) * lambda_constantine[0:idx_gap + 1]

    # todo: check which implementation performs better
    # np.testing.assert_allclose(activity_scores, activity_scores_constantine.flatten())

    approx_model = ApproxModel(W1=w_active, model=test_model)

    # %% bootstrapping
    distance_bootstrapping, lambda_i_eig, distance_subspace = \
        bootstrapping(values_grad_f, M_boot, n_samples, w_active, idx_gap, dim, w_vec)

    W1_pinv = np.linalg.pinv(w_active)
    W1_pinv_test = np.matmul(np.linalg.inv(np.matmul(np.transpose(w_active), w_active)), np.transpose(w_active))
    # W1_pinv_right = np.matmul(np.transpose(w_active), np.linalg.inv(np.matmul(w_active, np.transpose(w_active))))
    # print(np.matmul(W1_pinv, w_active))

    y = -1
    x = w_active * y
    transform_coordinates_from_unit(x_lower, x_upper, x)

    y = 1
    x = w_active * y
    transform_coordinates_from_unit(x_lower, x_upper, x)

    # %% activity scores
    if bool_print:
        print('** ........................... Activity scores ........................... **')

    # save results
    if bool_print:
        print('** ........................... Save results ........................... **')

    if bool_save_data and not bool_gradient:
        file_gradient = os.path.join(data_saver.get_path_to_files(), 'gradient_approximation.txt')
        open(file_gradient, 'w+')

    max_rel_error_eig = None
    if test_model.analytic_solution_available:
        # print results
        # compare eigenvalues of \hat C with true eigenvalue of C
        true_eig_C = test_model.get_eigenvalues_C(rho)
        max_rel_error_eig = relative_error(true_value=true_eig_C, approx_value=lambda_eig)
        if bool_print:
            print("** Max relative error in eigenvalues")
            print(max_rel_error_eig)
        calc_distance_subspace(distance_subspace, true_distance, bool_print)

    if data_saver is not None:
        path2results = data_saver.get_path_to_files()

    return max_rel_error_eig, error_c_hat, activity_scores, true_activity_scores, n, path2results, \
           n_samples, lambda_eig, w_active, test_y, lambda_eig_true, idx_gap, idx_gap_true, distance_subspace, true_distance
