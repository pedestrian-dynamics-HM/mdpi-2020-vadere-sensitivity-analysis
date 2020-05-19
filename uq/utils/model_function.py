import pickle
import logging
import os
import warnings
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import RandomState

from builtins import any as b_any, list
from typing import Union

from uq.utils.DataSaver import DataSaver
from uq.utils.datatype import is_scalar, is_row_vector, is_vector, is_matrix
from uq.utils.datatype import box1d, unbox, get_dimension, box, box_to_n_dim
from uq.utils.file_io import check_scenario_file

from suqc import QuickVaryScenario, VadereConsoleWrapper

INTEGER_PARAMETERS = ["stepCircleResolution", "fixedSeed", "numberOfCircles", "spawnNumber", "fixedSeed"]
LIST_PARAMETERS = ["distributionParameters"]
GROUP_PARAMETERS = ["groupSizeDistribution"]
TOPOGRAPHY_SYMMETRY_PARAMETERS = ["bottleneck_width"]

LOWEST_SEED = 0
HIGHEST_SEED = 2 ** 31 - 1


class Model:

    def __init__(self, key: str, qoi: str, map_of_results=None, bool_stochastic_model: bool = False,
                 data_saver: DataSaver = None, analytic_solution_available: bool = False):
        if type(key) == str:  # otherwise, length of str is dimension
            key = [key]
        self.key = key
        self.qoi = qoi
        self.map_of_results = map_of_results
        if map_of_results is None:
            self.map_of_results = dict()
        self.computation_time = dict()
        self.analytic_solution_available = analytic_solution_available
        self.data_saver = data_saver
        self.stochastic_model = bool_stochastic_model

    def get_key(self) -> Union[str, list]:
        return self.key

    def get_stochastic_model(self) -> bool:
        return self.stochastic_model

    def get_qoi(self) -> Union[str, list]:
        return self.qoi

    def get_analytic_solution_available(self) -> bool:
        return self.analytic_solution_available

    def get_computation_time(self) -> float:
        return self.computation_time

    def get_map_results(self) -> dict:
        return self.map_of_results

    def get_dimension(self) -> int:
        return len(self.get_key())

    def set_data_saver(self, data_saver: DataSaver) -> None:
        self.data_saver = data_saver

    def get_data_saver(self) -> DataSaver:
        return self.data_saver

    def eval_model(self, parameter_value: Union[float, np.ndarray], random_state: RandomState = None) -> Union[
        float, np.ndarray]:
        # logger = logging.getLogger("Model.eval_model")
        # logger.info(parameter_value) -> too much output
        pass

    # todo: check syntax!
    def eval_model_averaged(self, parameter_value: Union[float, np.ndarray], no_runs_averaged: int = 1,
                            random_state: RandomState = None):
        # logger = logging.getLogger("Model.eval_model_averaged")
        # logger.info(parameter_value)  # -> too much output
        if no_runs_averaged == 1 or not self.get_stochastic_model():
            model_eval = self.eval_model(parameter_value, random_state)
            if np.ndim(model_eval) == 0:
                model_eval = box(model_eval)
        else:
            raise NotImplementedError("eval_model_averaged not implemented for this model type!")

        return model_eval, None, None

    def eval_data_min_model(self, parameter_value: Union[float, np.ndarray], data: Union[float, np.ndarray]) -> Union[
        float, np.ndarray]:
        return data - self.eval_model(parameter_value)

    def dump_to_file(self, folder: str) -> None:
        with open(os.path.join(folder, "model_config.pickle"), 'wb') as file:
            pickle.dump(self, file)
        with open(os.path.join(folder, "model_eval.pickle"), 'wb') as file:
            pickle.dump(self.map_of_results, file)

    # Function to approximate the gradient with finite differences
    # model_evac: Already known model evaluations - used for forward / backward difference
    # at the moment, the central difference is used
    def approximate_gradient(self, input_vector, step_size, model_eval_vec=None, n_runs_av: int = 1,
                             random_state: RandomState = None) -> np.ndarray:
        if np.ndim(input_vector) == 0:
            input_vector = box(input_vector)
        if is_scalar(step_size):
            step_size = np.ones(self.get_dimension()) * step_size
        if is_row_vector(input_vector):
            input_vector = np.expand_dims(input_vector, axis=1)
            warnings.warn('Input to approximate_gradient should be a column vector, not a row vector.')
        n_dim = np.size(input_vector, axis=0)

        n_samples = np.size(input_vector, axis=1)
        approx_gradient = np.ones(shape=(n_dim, n_samples))
        approx_gradient[:] = np.nan

        # todo: parallelize
        for j in range(0, n_samples):
            for i in range(0, n_dim):
                h_vec = np.zeros(shape=n_dim)
                h_vec[i] = step_size[i]
                value_plus_h = np.expand_dims(input_vector[:, j] + h_vec, axis=1)
                value_min_h = np.expand_dims(input_vector[:, j] - h_vec, axis=1)

                # todo collect config, run later
                model_eval_val_plus_h, _, _ = self.eval_model_averaged(value_plus_h, n_runs_av,
                                                                       random_state=random_state)
                model_eval_val_min_h, _, _ = self.eval_model_averaged(value_min_h, n_runs_av, random_state=random_state)

                # central difference
                tmp = (model_eval_val_plus_h - model_eval_val_min_h) / (2 * step_size[i])

                # forward difference
                # h_vec[i] = step_size
                # value_plus_h = np.expand_dims(input_vector[:, j]+h_vec, axis=1)
                # value = np.expand_dims(input_vector[:, j], axis=1)
                # model_eval_val = self.eval_model(value)
                # model_eval_val_plus_h = self.eval_model(value_plus_h)
                # tmp2 = (model_eval_val_plus_h - model_eval_vec[j]) / step_size (forward diff)

                approx_gradient[i, j] = tmp
        return approx_gradient


class IshigamiFunction(Model):
    # https://uqworld.org/t/ishigami-function/55

    def __init__(self, a, b):
        key = ["x_1", "x_2", "x_3"]
        qoi = "Response"
        self.a = a
        self.b = b
        self.dim = 3
        super(IshigamiFunction, self).__init__(key, qoi, None, bool_stochastic_model=False,
                                               analytic_solution_available=True)

    def eval_model(self, parameter_value: Union[float, np.ndarray], *random_state) -> np.ndarray:
        if is_vector(parameter_value):
            x1 = parameter_value[0]
            x2 = parameter_value[1]
            x3 = parameter_value[2]
        elif is_matrix(parameter_value):
            x1 = parameter_value[0, :]
            x2 = parameter_value[1, :]
            x3 = parameter_value[2, :]

        y = np.sin(x1) + self.a * np.square(np.sin(x2)) + self.b * np.power(x3, 4) * np.sin(x1)

        return y

    def get_true_first_order_indices(self) -> np.ndarray:
        v_1 = 0.5 * np.square(1 + (self.b * np.power(np.pi, 4)) / 5)
        d_1 = 1/2 + self.b*np.power(np.pi, 4)/5 + np.square(self.b)*np.power(np.pi,8)/50
        v_2 = np.square(self.a) / 8
        v_3 = 0
        s1 = v_1 / self.get_true_vy()
        s2 = v_2 / self.get_true_vy()
        s3 = v_3 / self.get_true_vy()
        return np.array([s1, s2, s3])

    def get_true_vy(self):
        # sobol-1999
        v_y = 1/2 + np.square(self.a) / 8 + self.b * np.power(np.pi, 4) / 5 + np.square(self.b) * np.power(np.pi,
                                                                                                     8) / 18
        return v_y

    def get_true_total_effect_indices(self) -> np.ndarray:
        d_1 = 1/2 + self.b*np.power(np.pi, 4)/5 + np.square(self.b)*np.power(np.pi,8)/50
        d_2 = np.square(self.a) / 8
        d_13 = (1/18 - 1/50)*np.square(self.b)*np.power(np.pi,8)

        d_tot1 = (d_1 + d_13)/self.get_true_vy()
        d_tot2 = (d_2)/self.get_true_vy()
        d_tot3 = (d_13)/self.get_true_vy()


        tmp = (8 * np.square(self.b) * np.power(np.pi, 8)) / 225
        v_t1 = 0.5 * np.square(1 + (self.b * np.power(np.pi, 4) / 5)) + tmp
        v_t2 = np.square(self.a) / 8
        v_t3 = tmp
        s_t1 = v_t1 / self.get_true_vy()
        s_t2 = v_t2 / self.get_true_vy()
        s_t3 = v_t3 / self.get_true_vy()


        return np.array([s_t1, s_t2, s_t3])


class CircuitModel(Model):
    from uq.utils.prior_distribution import UniformGenMult

    def __init__(self):
        key = ["R_b1", "R_b2", "R_f", "R_c1", "R_c2", "beta"]
        qoi = "Response"
        self.dim = 6
        super(CircuitModel, self).__init__(key, qoi, None, bool_stochastic_model=False,
                                           analytic_solution_available=True)

    def eval_model(self, parameter_value: Union[float, np.ndarray], *random_state) -> np.ndarray:
        # example from constantine-2017: http://www.sfu.ca/~ssurjano/otlcircuit.html
        if is_vector(parameter_value):
            R_b1 = parameter_value[0]
            R_b2 = parameter_value[1]
            R_f = parameter_value[2]
            R_c1 = parameter_value[3]
            R_c2 = parameter_value[4]
            beta = parameter_value[5]
        elif is_matrix(parameter_value):
            R_b1 = parameter_value[0, :]
            R_b2 = parameter_value[1, :]
            R_f = parameter_value[2, :]
            R_c1 = parameter_value[3, :]
            R_c2 = parameter_value[4, :]
            beta = parameter_value[5, :]
        else:
            raise Warning('Dimension of parameter value does not fit the model')

        V_b1 = 12 * R_b2 / (R_b1 + R_b2)
        term1 = (V_b1 + 0.74) * beta * (R_c2 + 9) / (beta * (R_c2 + 9) + R_f)
        term2 = 11.35 * R_f / (beta * (R_c2 + 9) + R_f)
        term3 = 0.74 * R_f * beta * (R_c2 + 9) / ((beta * (R_c2 + 9) + R_f) * R_c1)
        V_m = term1 + term2 + term3

        return V_m

    def get_C_matrix(self, rho: UniformGenMult) -> np.ndarray:
        # calculated with Gauss Quadrature (not exact truth) - from Matlab code of constantine-2017

        C = np.array([[2.4555075012e+00, -1.8840695197e+00, -7.5922850961e-01, 4.0379535157e-01, 4.8953133700e-04,
                       1.1616907862e-02], [
                          -1.8840695197e+00, 1.7038668418e+00, 6.7105606089e-01, -3.5661734586e-01, -4.6881645905e-04,
                          -1.1125308912e-02], [
                          -7.5922850961e-01, 6.7105606089e-01, 2.9015818078e-01, -1.6103873334e-01, -1.9146596624e-04,
                          -4.5593065768e-03], [
                          4.0379535157e-01, -3.5661734586e-01, -1.6103873334e-01, 1.0904998348e-01, 1.1583772359e-04,
                          2.7484719868e-03], [
                          4.8953133700e-04, -4.6881645905e-04, -1.9146596624e-04, 1.1583772359e-04, 2.0373109646e-07,
                          6.0499138189e-06], [
                          1.1616907862e-02, -1.1125308912e-02, -4.5593065768e-03, 2.7484719868e-03, 6.0499138189e-06,
                          2.1028328072e-04]])
        return C

    def get_eigenvalues_C(self, rho: UniformGenMult) -> np.ndarray:
        # calculated with Gauss Quadrature (not exact truth) - from Matlab code of constantine-2017
        lambda_C = np.array([4.3339297734, 0.1721545468, 0.0438372806, 0.0087407672, 0.0001306198, 0.0000000066])
        return lambda_C

    def eval_gradient(self, value_input: np.ndarray) -> np.ndarray:

        if is_vector(value_input):
            Rb1 = value_input[0]
            Rb2 = value_input[1]
            Rf = value_input[2]
            Rc1 = value_input[3]
            Rc2 = value_input[4]
            beta = value_input[5]
        elif is_matrix(value_input):
            Rb1 = value_input[0, :]
            Rb2 = value_input[1, :]
            Rf = value_input[2, :]
            Rc1 = value_input[3, :]
            Rc2 = value_input[4, :]
            beta = value_input[5, :]
        else:
            raise Warning('Dimension of value_input does not fit the model')
        n_values = int(np.size(value_input) / self.get_dimension())
        dV = np.zeros(shape=(self.get_dimension(), n_values))

        dV[0, :] = (-12 * Rb2 * beta * (Rc2 + 9)) / ((beta * (Rc2 + 9) + Rf) * np.square(Rb1 + Rb2))

        dV[1, :] = (12 * Rb1 * beta * (Rc2 + 9)) / ((beta * (Rc2 + 9) + Rf) * np.square(Rb1 + Rb2))

        dV[2, :] = (beta * (beta * (Rb1 + Rb2) * (59.94 + 13.32 * Rc2 + 0.74 * np.square(Rc2)) + Rc1 * (
                Rb2 * (-12.51 - 1.39 * Rc2) + Rb1 * (95.49 + 10.61 * Rc2)))) / (
                           (Rb1 + Rb2) * Rc1 * np.square(beta * (9 + Rc2) + Rf))

        dV[3, :] = -(0.74 * beta * (Rc2 + 9) * Rf) / (np.square(Rc1) * (beta * (Rc2 + 9) + Rf))

        dV[4, :] = (beta * Rf * (-10.61 * Rb1 * Rc1 + 1.39 * Rb2 * Rc1 + 0.74 * Rb1 * Rf + 0.74 * Rb2 * Rf)) / (
                (Rb1 + Rb2) * Rc1 * np.square(beta * (9 + Rc2) + Rf))

        dV[5, :] = (Rf * (Rb1 * (-95.49 * Rc1 - 10.61 * Rc1 * Rc2 + 6.66 * Rf + 0.74 * Rc2 * Rf) + Rb2 * (
                12.51 * Rc1 + 1.39 * Rc1 * Rc2 + 6.66 * Rf + 0.74 * Rc2 * Rf))) / (
                           (Rb1 + Rb2) * Rc1 * np.square(beta * (9 + Rc2) + Rf))

        return dV


class ExponentialModel(Model):
    # from constantine-2015
    from uq.utils.prior_distribution import UniformGenMult

    def __init__(self):
        # self.key = ["x1", "x2"]
        # self.qoi = ""
        self.dim = 2
        super(ExponentialModel, self).__init__(["x1", "x2"], "f", None, bool_stochastic_model=True,
                                               analytic_solution_available=True)

    def get_dimension(self) -> int:
        return self.dim

    def get_C_matrix(self, rho: UniformGenMult) -> np.ndarray:
        # only valid for uniform distribution [-1,1]^2
        factor = np.sum(rho.upper - rho.lower)
        result = 1 / factor * 1 / 84 * np.array([[49, 21], [21, 9]]) * \
                 (self.eval_model(rho.upper) ** 2 - self.eval_model(np.array([rho.lower[0], rho.upper[1]])) ** 2 -
                  self.eval_model(np.array([rho.upper[0], rho.lower[1]])) ** 2 + self.eval_model(rho.lower) ** 2)
        # C_mat = np.array([[0.707222, 0.303095], [0.303095, 0.129898]])
        return result

    def get_eigenvalues_C(self, rho: UniformGenMult) -> np.ndarray:
        # factor = np.sum(rho.upper - rho.lower)
        C_mat = self.get_C_matrix(rho)
        eig_C = np.linalg.eigvals(C_mat)
        return eig_C

    def eval_model(self, parameter_value: np.ndarray, *random_state) -> float:
        if is_vector(parameter_value) and parameter_value.size == self.get_dimension():
            if np.ndim(parameter_value) > 1:
                parameter_value = unbox(parameter_value)
            # for i in range(0, value_input.shape[1]):
            # y = np.matmul(self.W1, np.transpose(self.W1) * value_input[:,i])
            f_value = np.exp(0.7 * parameter_value[0] + 0.3 * parameter_value[1])
        elif is_matrix(parameter_value) and np.mod(np.size(parameter_value), self.get_dimension()) == 0:
            if np.size(parameter_value, axis=1) is self.get_dimension() and not (
                    np.size(parameter_value, axis=0) == self.get_dimension()):
                warnings.warn("TestModel.eval_model(): Dimensions of input do not fit. Input was transposed.")
                np.transpose(parameter_value)
            f_value = np.exp(0.7 * parameter_value[0, :] + 0.3 * parameter_value[1, :])
        else:
            raise Warning("eval_model: Input does not match the model")
        return f_value

    def eval_gradient(self, value_input: np.ndarray) -> np.ndarray:
        if is_vector(value_input):
            n_values = int(value_input.size / self.get_dimension())
            value_input = box(value_input)
        elif is_matrix(value_input):
            n_values = np.size(value_input, 1)
        else:
            raise Warning("Eval gradient: Input does not have the expected dimensions")

        for i in range(0, n_values):
            # partial derivatives
            f_x1 = np.exp(0.7 * value_input[0, i] + 0.3 * value_input[1, i]) * 0.7
            f_x2 = np.exp(0.7 * value_input[0, i] + 0.3 * value_input[1, i]) * 0.3
            f_value = np.array([[f_x1], [f_x2]])
            if i == 0:
                gradient = f_value
            else:
                gradient = np.append(gradient, f_value, axis=1)
        return gradient


class QuadraticModel(Model):
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b
        self.dim = 1
        super(QuadraticModel, self).__init__("x", "f", None, bool_stochastic_model=False,
                                             analytic_solution_available=True)

    def get_dimension(self) -> int:
        return self.dim

    def eval_model(self, parameter_value: float, *random_state) -> float:
        return self.a * np.square(parameter_value) + self.b

    def eval_gradient(self, parameter_value: float) -> float:
        return 2 * self.a * parameter_value


class MatrixModel(Model):
    from uq.utils.prior_distribution import UniformGenMult

    def __init__(self, A: np.ndarray):
        self.A = A
        self.eig = None
        self.dim = np.size(A, axis=0)
        super(MatrixModel, self).__init__(" ", " ", None, bool_stochastic_model=False, analytic_solution_available=True)

    def get_dimension(self) -> int:
        return np.size(self.A, axis=1)

    def set_eigenvalues(self, eig: np.ndarray) -> None:
        self.eig = eig

    def get_eigenvalues(self) -> np.ndarray:
        return self.eig

    def get_eigenvalues_C(self, rho: "UniformGenMult") -> np.ndarray:

        # if type(rho) is UniformGenMult:
        eig = 1 / 3 * (self.eig ** 2)
        # else:
        #    raise Warning('C-Matrix not implemented for this distribution')
        return eig

    def get_C_matrix(self, rho: UniformGenMult) -> np.ndarray:
        # only valid for uniform distribution [-1,1]^m
        # if type(rho) is UniformGenMult:
        C = 1 / 3 * np.matmul(self.A, self.A)
        # else:
        #    raise Warning('C-Matrix not implemented for this distribution')
        return C

    def eval_model(self, parameter_value: np.ndarray, *random_state) -> float:
        if is_scalar(parameter_value):
            result = 1 / 2 * parameter_value * self.A * parameter_value
        else:
            if is_row_vector(parameter_value):
                parameter_value = np.expand_dims(parameter_value, axis=1)

            m = np.size(parameter_value, axis=1)
            result = np.zeros(shape=m)
            for i_cols in range(0, m):
                result[i_cols] = 1 / 2 * np.matmul(np.matmul(np.transpose(parameter_value[:, i_cols]), self.A),
                                                   parameter_value[:, i_cols])
        return result

    def eval_gradient(self, parameter_value: np.ndarray) -> np.ndarray:
        if is_scalar(parameter_value):
            gradient = self.A * parameter_value
        else:
            gradient = np.matmul(self.A, parameter_value)
        return gradient


class ApproxModel(Model):

    def __init__(self, W1: Union[float, np.ndarray], model: Model):
        self.key = ""
        self.qoi = ""
        self.dim = 2
        self.W1 = W1
        self.model = model
        super(ApproxModel, self).__init__(" ", " ", None, bool_stochastic_model=False,
                                          analytic_solution_available=False)

    def eval_model(self, parameter_value: Union[float, np.ndarray], *random_state) -> Union[float, np.ndarray]:
        if np.size(parameter_value) == 1:
            input_value = self.W1 * parameter_value
        else:
            input_value = np.matmul(self.W1, parameter_value)
        g = self.model.eval_model(input_value)
        return g


class VadereModel(Model):

    def __init__(self, run_local: bool, path2scenario: str, path2model: str, key: Union[str, list],
                 qoi: Union[str, list], n_jobs: int, log_lvl: str):
        self.run_local = run_local
        self.path2scenario = path2scenario
        self.path2model = path2model
        self.n_jobs = n_jobs
        self.log_lvl = log_lvl
        self.n_evals = 0

        check_scenario_file(path2scenario)

        scenario_file_hash = self.hash_scenario_file(path2scenario)
        map_of_results = dict({"scenario_file_hash": scenario_file_hash})
        super(VadereModel, self).__init__(key, qoi, map_of_results, bool_stochastic_model=True,
                                          analytic_solution_available=False)

    def get_run_local(self) -> bool:
        return self.run_local

    def get_path_2_scenario(self) -> str:
        return self.path2scenario

    def get_path_2_model(self) -> str:
        return self.path2model

    def get_loglvl(self) -> str:
        return self.log_lvl

    def get_n_jobs(self) -> int:
        return self.n_jobs

    def set_key(self, in_key) -> Union[str, list]:
        self.key = in_key

    def get_n_evals(self) -> int:
        return self.n_evals

    def set_n_evals(self, n: int) -> None:
        self.n_evals = n

    def set_run_local(self, run_local: bool) -> None:
        self.run_local = run_local

    def get_dimension(self) -> int:
        if type(self.get_key()) == str:  # otherwise, length of str is dimension
            dim = 1
            raise Warning('Type of key should be list and not just a string -> look at constructor in Model')
        else:
            dim = len(self.get_key())
        return dim

    @staticmethod
    def hash_scenario_file(path2scenario: str) -> int:
        with open(path2scenario, 'rb') as file:
            return hash(file)

    @staticmethod
    def add_seed(n: int, value_input, key: Union[str, list], random_state: RandomState = None):
        if random_state is None:
            warnings.warn("VadereModel.add_seed(): random_state is empty!")
            random_state = RandomState()

        # generate seeds
        seeds = np.array([random_state.rand(n) * (HIGHEST_SEED - LOWEST_SEED) + LOWEST_SEED]).astype(int)
        if type(key) == str or type(key) == list:  # only one key is varied
            if type(value_input) == np.ndarray:  #
                ext_value_input = np.concatenate((box_to_n_dim(value_input, 2), seeds), axis=0)
                new_key = key + list(["fixedSeed"])  # assure that self.key is not changed (key.append changes self.key)
            else:
                raise Exception("Unexpected type of value_input: %s" % type(value_input))
        else:
            raise Exception("Unexpected type of key: %s" % type(key))

        return ext_value_input, new_key, seeds

    def create_dict_list(self, n: int, value_input) -> list:

        dict_list = self.create_dict_list_with_key(n=n, value_input=value_input, in_key=self.get_key())

        return dict_list

    @staticmethod
    def create_dict_list_with_key(n: int, value_input: np.ndarray, in_key: Union[str, list]) -> list:
        dict_list = []
        for i in range(0, n):  # iterate over samples
            param_dict = dict()
            for j in range(0, get_dimension(in_key)):  # iterate over input parameters
                # if key is nested key, extract just the last bit to find the type
                tmp_parts = str.split(in_key[j], '.')

                bool_set_key = True

                if len(tmp_parts) > 1:
                    key = tmp_parts[-1]  # last entry
                else:
                    key = in_key[j]

                if b_any(key in x for x in INTEGER_PARAMETERS):  # self.key[j] in INTEGER_PARAMETERS:
                    value = int(np.round(value_input[j, i]))
                elif b_any(key in x for x in LIST_PARAMETERS):
                    value = [value_input[j, i]]  # convert to list
                elif b_any(key in x for x in GROUP_PARAMETERS):
                    value = [1 - value_input[j, i],
                             1 - value_input[j, i]]  # convert to list with percentage of singles and couples
                elif b_any(key in x for x in TOPOGRAPHY_SYMMETRY_PARAMETERS):
                    # todo: special file or sth for this config that is only for one specific scenario file
                    # vertical center: 7.5
                    v_center_line = 7.5
                    obstacle_height = 6.0
                    my_key = "obstacles.[id==1].y"  # lower obstacle (Liddle_bhm_v3)
                    my_value = -value_input[j, i] / 2 + v_center_line - obstacle_height
                    param_dict[my_key] = my_value

                    my_key = "obstacles.[id==2].y"  # upper obstacle
                    my_value = value_input[j, i] / 2 + v_center_line
                    param_dict[my_key] = my_value

                    distance_to_obstacle = 0.3  # distance between intermediate target and obstacle
                    height_of_obstacle = 10

                    # for intermediate target
                    # my_key = "targets.[id==1].height"
                    # my_value = value_input[j, i]
                    # target_1_height = my_value - distance_to_obstacle * 2
                    # param_dict[my_key] = target_1_height
                    #
                    # my_key = "targets.[id==1].y"  # intermediate target
                    # my_value = v_center_line - target_1_height / 2
                    # param_dict[my_key] = my_value

                    my_key = "targets.[id==1].height"
                    target_2_height = value_input[j, i]
                    param_dict[my_key] = target_2_height

                    my_key = "targets.[id==1].y"  # target
                    my_value = v_center_line - target_2_height / 2
                    param_dict[my_key] = my_value

                    bool_set_key = False

                else:
                    value = value_input[j, i]

                if bool_set_key:
                    param_dict[in_key[j]] = value

            dict_list.append(param_dict)
        return dict_list

    def evaluate_deviation_from_mean(self, results_reshaped: np.ndarray, result_averaged: np.ndarray,
                                     value_input: Union[float, np.ndarray], no_points_averaged: int = 1) -> np.ndarray:
        dev = results_reshaped - np.tile(np.expand_dims(result_averaged, axis=1), (1, no_points_averaged))
        if np.ndim(value_input) == 1:
            plt.figure()
            value_input_formatted = np.expand_dims(value_input, axis=1)
            plt.plot(np.tile(value_input_formatted, reps=(1, no_points_averaged)), dev, marker='.')

            # n_param_values = np.size(dev, axis=0) + 1
            # ax1 = plt.subplot(n_param_values, 1, 1)
            for i in range(0, np.size(dev, axis=0)):
                h3 = plt.figure()
                # plt.subplot(n_param_values, 1, i + 1).\
                plt.hist(dev[i, :], stacked=False, label=str(value_input[i]), density=True)
                # plt.xlim([-0.1, 0.1])
                plt.title(str(value_input[i]))
                if self.get_data_saver() is not None:
                    self.get_data_saver().save_figure(h3, ("deviations_over_parameter%d" % i))

        # else:
        # value_input_formatted = value_input

        h1 = plt.figure()
        plt.hist(np.ravel(dev), 100, density='true')
        mean_est = np.mean(np.ravel(dev))
        std_est = np.std(np.ravel(dev))
        x = np.linspace(np.min(np.ravel(dev)), np.max(np.ravel(dev)))
        random_normal = stats.norm(mean_est, std_est).pdf(x)
        plt.plot(x, random_normal, '--r', label='Fitted normal distribution')
        plt.legend()
        if not self.get_data_saver() is None:
            self.get_data_saver().save_figure(h1, "histogram_deviations_vadere")
        # plt.savefig('histogram_deviations_vadere.png')

        sm.qqplot(np.ravel(dev), line='s')
        h2 = plt.gcf()
        # plt.savefig('qqplot_deviations_vadere.png')
        if not self.get_data_saver() is None:
            self.get_data_saver().save_figure(h2, "qqplot_deviations_vadere")

        plt.close(h2)

        vadere_logger = logging.getLogger("vaderemodel.evaluate_deviation_from_mean")
        vadere_logger.info("Vadere evaluations: Deviations from average")
        vadere_logger.info("Mean: %f, Std: %f" % (mean_est, std_est))

        # skewtest needs at least 8 samples
        if len(np.ravel(dev)) >= 20:
            alpha = 0.01
            k2, p = stats.normaltest(np.ravel(dev))
            vadere_logger.info("p = {:g}".format(p))
            if p < alpha:  # null hypothesis: x comes from a normal distribution
                vadere_logger.info("The null hypothesis can be rejected")
            else:
                vadere_logger.info("The null hypothesis cannot be rejected")

        return dev

    def eval_model_multiple_times(self, parameter_value: Union[float, np.ndarray], no_runs: int = 1,
                                  bool_fixed_seed: bool = False, random_state: RandomState = None):
        if np.mod(len(box1d(parameter_value)), self.get_dimension()) > 0:
            raise Warning("Shapes of value_input and size of key not compatible")

        if self.get_dimension() == 1:
            number_of_values = np.size(box1d(parameter_value), axis=0)
        else:
            # todo: check if this works for multi-dim input
            number_of_values = np.size(box(parameter_value), axis=1)
            # if number_of_values == self.get_dimension() and np.size(box1d(value_input), axis=0):
            #     raise Warning("Dimension of value_input does not fit the expectations. Might need to be transposed")

        loglvl = self.get_loglvl()

        if no_runs > 1:
            value_input_duplicated = np.ravel(np.transpose(np.tile(parameter_value, reps=(no_runs, 1))))
        else:
            value_input_duplicated = parameter_value

        if type(value_input_duplicated) == float or type(value_input_duplicated) == int or type(
                value_input_duplicated) == np.float64:
            value_input_duplicated = np.ones(shape=(1, 1)) * value_input_duplicated
        elif type(value_input_duplicated) == np.ndarray:
            if np.ndim(value_input_duplicated) == 1:
                value_input_duplicated = np.expand_dims(value_input_duplicated, axis=0)
                value_input_duplicated = np.transpose(
                    np.reshape(value_input_duplicated, newshape=(-1, self.get_dimension())))
            elif no_runs > 1:
                raise Exception("More-dimensional version (with multiple runs) not yet implemented")

        # n = np.size(value_input_duplicated, axis=1)

        #if self.get_dimension() == 1:
        #    warnings.warn("VadereModel.eval_model_multiple_times(): single_eval_model -> Seed cannot be supplied")
        #    result = self.single_eval_model(value_input_duplicated, loglvl, n=no_runs * number_of_values,
        #                                    bool_fixed_seed=bool_fixed_seed, random_state=random_state)
        #    new_key = self.get_key()#
        #
        #    ext_value_input = value_input_duplicated

        #else:
        # create dict
        if not bool_fixed_seed:
            ext_value_input, new_key, seeds = self.add_seed(n=no_runs * number_of_values,
                                                            value_input=value_input_duplicated, key=self.get_key(),
                                                            random_state=random_state)
        else:
            new_key = self.get_key()
            ext_value_input = value_input_duplicated

        dict_list = self.create_dict_list_with_key(n=no_runs * number_of_values, value_input=ext_value_input,
                                                   in_key=new_key)

        # dict_list = self.create_dict_list(n=no_runs, value_input=value_input_duplicated)  #without seed
        result = self.single_eval_model(dict_list, loglvl,
                                        n=no_runs * number_of_values)  # eval model at list of dicts

        data_saver = self.get_data_saver()
        if data_saver is not None:
            self.get_data_saver().write_model_eval_to_file(key=new_key, qoi=self.get_qoi(), parameters=dict_list,
                                                           results=result)

        return result, ext_value_input

    # Evaluate an array of sample points and average for each point
    def eval_model_averaged(self, parameter_value: Union[float, np.ndarray], no_runs_averaged: int = 1,
                            bool_fixed_seed: bool = False, random_state: RandomState = None, bool_plot: bool = False):

        result, value_input_duplicated = self.eval_model_multiple_times(parameter_value, no_runs_averaged,
                                                                        bool_fixed_seed, random_state)

        # average results
        if no_runs_averaged > 1:

            results_reshaped = np.reshape(np.array(result), (-1, no_runs_averaged))

            result_averaged = np.mean(results_reshaped, axis=1)

        else:
            if type(result) is not dict and type(result) is not pd.DataFrame:
                result_averaged = np.array(result)
            else:
                result_averaged = result

        return result_averaged, value_input_duplicated, result

    def add_n_evals(self, n: int) -> None:
        self.set_n_evals(self.get_n_evals() + n)

    def eval_model(self, parameter_value: Union[float, np.ndarray], bool_fixed_seed: bool = False,
                   random_state: RandomState = None) -> Union[float, np.ndarray]:
        # Evaluate an array of sample points
        # model_eval, all_input, all_output = self.eval_model_averaged(value_input, no_points_averaged=1)

        # instead of overloading method
        model_eval, all_input, all_output = self.eval_model_averaged(parameter_value, no_runs_averaged=1,
                                                                     bool_fixed_seed=bool_fixed_seed,
                                                                     random_state=random_state)
        return model_eval

    def single_eval_model(self, value_input: Union[float, np.ndarray], loglvl_in: str, n: int = 1,
                          bool_fixed_seed: bool = False, random_state: RandomState = None) -> Union[float, np.ndarray]:

        logger = logging.getLogger("VadereModel.single_eval_model")
        value = box1d(value_input)

        vadere_model = VadereConsoleWrapper(model_path=self.get_path_2_model(), loglvl=loglvl_in)

        if type(value) == np.ndarray:
            key = self.get_key()

            # add seed to run
            if not bool_fixed_seed:
                value, key, seeds = self.add_seed(n=n, value_input=value, key=self.get_key(), random_state=random_state)

            value = self.create_dict_list_with_key(n=n, value_input=value, in_key=key)

            setup_scenario = QuickVaryScenario(scenario_path=self.get_path_2_scenario(),
                                               parameter_var=value,
                                               qoi=self.get_qoi(),
                                               model=vadere_model)

            """setup_scenario = SingleKeyVaryScenario(scenario_path=self.get_path_2_scenario(),
                                                   # -> path to the Vadere .scenario file to vary
                                                   key=key,  # -> parameter key to change
                                                   values=value,  # -> values to set for the parameter
                                                   qoi=self.get_qoi(),  # -> output file name to collect
                                                   model=vadere_model
                                                   # -> path to Vadere console jar file to use for simulation
                                                   )"""

        else:
            if type(value) == list:
                setup_scenario = QuickVaryScenario(scenario_path=self.get_path_2_scenario(),
                                                   parameter_var=value,
                                                   qoi=self.get_qoi(),
                                                   model=vadere_model)
                key = self.get_key()
            else:
                raise Exception("Type ist not allowed here %s " % type(value))

        if self.get_run_local():
            par_var, data = setup_scenario.run(self.get_n_jobs())
        else:
            par_var, data = setup_scenario.remote(self.get_n_jobs())

        self.add_n_evals(len(value))

        if data is None:
            logger.exception("VadereModel.single_eval_model: Resulting data of Simulation run is None")
            raise ValueError('Resulting data of Simulation run is None')

        if self.get_qoi() in ["evac_time.txt", "mean_density.txt", "waitingtime.txt", "flow.txt",
                              "cTimeStep.fundamentalDiagram", "max_density.txt"]:
            if self.get_qoi() == "evac_time.txt":
                identifier = "evacuationTime"
            elif self.get_qoi() == "mean_density.txt":
                identifier = "mean_area_density_voronoi_processor"
            elif self.get_qoi() == "max_density.txt":
                identifier = "max_area_density_voronoi_processor"
            elif self.get_qoi() == "waitingtime.txt":
                identifier = "waitingTimeStart"
            elif self.get_qoi() == "flow.txt":
                identifier = "flow"
            elif self.get_qoi() == "cTimeStep.fundamentalDiagram":
                # todo: handle > 1 identifier
                identifier = ["density", "velocity"]
            else:
                raise Exception("model_function.py: Identifier not yet defined")

            """  figure out the processor id '-PIDx' """
        if type(identifier) == str:
            identifier_list = [s for s in data.columns if identifier in s]

            if len(identifier_list) == 1:
                identifier = identifier_list[0]

                if type(value) == float:
                    result = data[identifier].values[0]
                else:
                    result = data[identifier]

            else:  # multiple entries in the file fit the identifier
                if identifier == "flow":

                    if type(value) == list:
                        result = data.values
                else:
                    logger.debug("model function: Warning: Identifier %s not found" % identifier)

        elif type(identifier) == list:  # multiple identifiers necessary for QOI eval
            result = dict()
            for j in range(0, len(identifier)):  # for each identifier
                identifier_list = [s for s in data.columns if
                                   identifier[j] in s]  # first entry with this identifier (to get the ID)
                result[identifier[j]] = data[identifier_list[0]]
            result = data

        else:
            logger.exception("model_function.py:: QOI not yet defined")
            raise Exception("model_function.py:: QOI not yet defined")

        """if type(value) == np.ndarray:     # only one parameter is varied and one variation
            if np.ndim(value) > 1:
                if np.size(value, axis=1) == 1:
                    while np.ndim(value) > 0:
                        value = value.item()
        
                    self.computation_time[value] = par_var["req_time"].values[0]
                    # add to dictionary with results
                    self.map_of_results[value] = result """

        # data_saver = self.get_data_saver()
        # if data_saver is not None:
        #    data_saver.write_model_eval_to_file(key=key, qoi=self.get_qoi(), parameters=value, results=result)
        # else:
        #    print("No datasaver assigned to model")

        return result


class Functions:
    def rational(self, x: float, p: float, q: float) -> float:
        """
        The general rational function description.
        p is a list with the polynomial coefficients in the numerator
        q is a list with the polynomial coefficients (except the first one)
        in the denominator
        The zeroth order coefficient of the denominator polynomial is fixed at 1.
        Numpy stores coefficients in [x**2 + x + 1] order, so the fixed
        zeroth order denominator coefficient must comes last. (Edited.)
        """
        return np.polyval(p, x) / np.polyval(q, x)

    def rational3_3(self, x: float, p0: float, p1: float, p2: float, q1: float, q2: float) -> float:
        return Functions.rational(self, x, [p0, p1, p2], [q1, q2])

    def rational1(self, x: float, p0: float, q1: float, q2: float) -> float:
        return Functions.rational(self, x, [p0], [q1, q2])

    def polynomial(self, x: float, p: float) -> float:
        return np.polyval(p, x)

    def polynomial0(self, x: float, p0: float) -> float:
        return Functions.polynomial(self, x, p0)

    def polynomial1(self, x: float, p0: float, p1: float) -> float:
        return Functions.polynomial(self, x, [p0, p1])

    def polynomial2(self, x: float, p0: float, p1: float, p2: float) -> float:
        return Functions.polynomial(self, x, [p0, p1, p2])
