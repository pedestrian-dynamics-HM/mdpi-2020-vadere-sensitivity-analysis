import numpy as np
from numpy.random import RandomState
from typing import Union
import warnings

from uq.utils.datatype import is_row_vector

MACHINE_PRECISION = np.finfo(float).eps


def relative_error_constantine_2017(estimation: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(estimation - reference) / np.max(np.abs(reference))


def transform_coordinates_to_unit(x_lower: Union[float, np.ndarray], x_upper: Union[float, np.ndarray],
                                  value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    if np.ndim(value) == 1 and len(value) > 1:  # row vector
        value = np.expand_dims(value, axis=1)  # shape to column vector
    return np.matmul(np.diag(1 / (x_upper - x_lower)), ((2 * value)) - np.expand_dims(x_upper + x_lower, axis=1))


# from constantine-2015, p. 2, eq. (1.1)
def transform_coordinates_from_unit(x_lower: Union[float, np.ndarray], x_upper: Union[float, np.ndarray],
                                    value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    if np.ndim(value) == 1 and len(value) > 1:  # row vector
        value = np.expand_dims(value, axis=1)  # shape to column vector
    if np.size(value) > 1:
        transformed_value = 1 / 2 * (
                np.matmul(np.diag(x_upper - x_lower), value) + np.expand_dims(x_upper + x_lower, axis=1))
    else:
        transformed_value = 1 / 2 * ((np.diag(x_upper - x_lower) * value) + np.expand_dims(x_upper + x_lower, axis=1))

    value_datatype = transformed_value
    # for i in range(0,np.size(value,axis=0)-1):
    #     if boolean_vec[i]:
    #         value_datatype[i,:] = value_datatype[i,:].astype(dtype=int)
    #         print(value_datatype)

    return value_datatype


def find_largest_gap_log(lambda_eig: np.ndarray, step_size: float = None):
    # make sure that the vector is sorted
    if any(np.diff(lambda_eig) > 0):
        raise ValueError("find_largest_gap_log: Input vector must be descending!")
    if (np.abs(lambda_eig) < -np.finfo(float).eps).any():  # lambda_eig should always be >= 0 since C is positive semi-definite, abs is for numerical instabilities
        idx = np.argmax(np.abs(np.diff(lambda_eig)))  # find largest gap in normal representation
        bool_plot_log = False
    else:
        if step_size is not None:  # gradients are approximated
            lambda_eig = lambda_eig[lambda_eig > step_size]  # only choose a gap that is above the step_size

        idx = np.argmax(np.abs(np.diff(np.log10(lambda_eig))))  # find largest gap in logarithmic representation
        bool_plot_log = True
    return idx, bool_plot_log


def relative_error(true_value: Union[float, np.ndarray], approx_value: Union[float, np.ndarray]) -> float:
    return (approx_value - true_value) / true_value


def abs_relative_error(true_value: Union[float, np.ndarray], approx_value: Union[float, np.ndarray]) -> float:
    return np.abs(approx_value - true_value) / true_value


def compute_C_from_gradients(values_grad_f: np.ndarray, M: int, m: int) -> np.ndarray:
    C_hat = construct_C_matrix(values_grad_f, M, m, False)
    return C_hat


def construct_C_matrix(values_grad_f: np.ndarray, M: int, m: int, bool_bootstrap: bool,
                       random_state: RandomState = None) -> np.ndarray:
    if random_state is None:
        random_state = RandomState()
    C_hat = np.zeros(shape=(m, m))
    for i in range(0, M):
        if bool_bootstrap:
            # draw a random integer j_k between 1 and M
            j_k = int(np.round(random_state.rand(1) * M - 0.5))
        else:
            j_k = i

        tmp = np.outer(values_grad_f[:, j_k], np.transpose(values_grad_f[:, j_k]))
        C_hat = C_hat + tmp

    C_hat = C_hat * 1 / M
    C_hat_tmp = 1 / M * np.matmul(values_grad_f, np.transpose(values_grad_f))

    if bool_bootstrap:
        C_hat_tmp = C_hat
    # todo: check why differences occur - even without bootstrap
    # todo: move to tests
    np.testing.assert_array_almost_equal(C_hat, C_hat_tmp)

    return C_hat_tmp


def bootstrapping(values_grad_f: np.ndarray, M_boot: int, M: int, w_active, subspace_dim: int, m: int,
                  w_vec: np.ndarray):
    distance_e = None
    distance_subspace = None
    lambda_eig_i_save = None

    if M_boot > 0:
        # todo: parallelize

        m = np.size(values_grad_f, axis=0)

        distance_e = np.zeros(shape=(m - 1, M_boot))
        lambda_eig_i_save = np.zeros(shape=(m, M_boot))

        for i in range(0, M_boot):

            C_hat_i = construct_C_matrix(values_grad_f, M, m, True)

            # Compute the eigendecomposition

            w_vec_i, lambda_eig_i, wh_i = eigendecomposition(C_hat_i)
            lambda_eig_i_save[:, i] = lambda_eig_i

            # For a particular choice of the active subspace dimension n [constantine-2015]
            for subspace_idx in range(0, m - 1):
                w_active_i, w_inactive_i = divide_active_inactive_subspace(w_vec_i, subspace_idx + 1, m)

                w_active, w_inactive = divide_active_inactive_subspace(w_vec, subspace_idx + 1, m)

                # distance between subspaces according to 3.73
                distance_e[subspace_idx, i] = distance_subspace_eq_349(w_active, w_inactive_i)
                # np.linalg.norm(np.matmul(np.transpose(w_active), w_inactive_i))

        distance_subspace = np.mean(distance_e, axis=1)

    return distance_e, lambda_eig_i_save, distance_subspace


def eigendecomposition(C_hat: np.ndarray):
    w_vec, lambda_eig, w_h = np.linalg.svd(C_hat)

    # todo: if linalg.eig is used, the eigenvalues (and vectors!) have to be sorted afterwards in descending order
    # lambda_eig, w_vec = np.linalg.eig(C_hat)
    w_h = np.transpose(w_vec)

    threshold = 1e-9
    max_diff = np.max(np.matmul(np.matmul(w_vec, np.diag(lambda_eig)), w_h) - C_hat)
    if max_diff > threshold:  # np.finfo(float).eps:
        warnings.warn("Eigendecomposition: Errors are greater than %.3e (%.3e)" % (threshold, max_diff))
    return w_vec, lambda_eig, w_h


def distance_subspace_corollary310(lambda_eig: np.ndarray, idx_gap: int, n_dim: int) -> float:
    if idx_gap == n_dim - 1:
        # distance cannot be calculated if the active subspace is the whole space
        dist_as = None
    else:
        eps = ((lambda_eig[idx_gap] - lambda_eig[idx_gap + 1]) / (5 * lambda_eig[0]))
        dist_as = (4 * lambda_eig[0] * eps) / (lambda_eig[idx_gap] - lambda_eig[idx_gap + 1])  # upper bound
    return dist_as


# provided by D. Lehmberg
def assert_allclose_eigenvectors(eigvec1: np.ndarray, eigvec2: np.ndarray, tol: float = 1e-14) -> None:
    # Allows to also check orthogonality, but is not yet implemented
    norms1 = np.linalg.norm(eigvec1, axis=0)
    norms2 = np.linalg.norm(eigvec2, axis=0)
    eigvec_test = (eigvec1.T @ eigvec2) * np.reciprocal(np.outer(norms1, norms2))

    actual = np.abs(np.diag(eigvec_test))  # -1 is also allowed for same direction
    expected = np.ones(actual.shape[0])

    np.testing.assert_allclose(expected, actual, atol=tol, rtol=0)


def distance_subspace_eq_349(W1: np.ndarray, W2_hat: np.ndarray) -> float:
    if is_row_vector(W1):  # W1 is row vector
        W1 = np.expand_dims(W1, axis=1)
    dist = np.linalg.norm(np.matmul(np.transpose(W1), W2_hat))
    return dist


def is_smaller_equal_than_machine_precision(a: float) -> bool:
    return abs(a) <= np.finfo(float).eps


def divide_active_inactive_subspace(w_vec: np.ndarray, subspace_dim: int, m: int):
    w_active_tmp = w_vec[:, 0:subspace_dim]
    w_inactive_tmp = w_vec[:, subspace_dim:m]
    return w_active_tmp, w_inactive_tmp


def least_squares(matrix_x: np.ndarray, y: np.ndarray) -> np.ndarray:
    tmp = np.linalg.inv(np.matmul(np.transpose(matrix_x), matrix_x))
    beta_hat = np.matmul(np.matmul(tmp, np.transpose(matrix_x)), y)
    return beta_hat


def calc_activity_scores(lambda_eig: np.array, w_vec: np.array, m: int, n: int) -> np.ndarray:
    alpha = np.zeros(shape=m)
    for i in range(0, m):
        tmp = 0
        for j in range(0, n):
            tmp = tmp + (lambda_eig[j] * np.square(w_vec[i, j]))

        alpha[i] = tmp
    return alpha




