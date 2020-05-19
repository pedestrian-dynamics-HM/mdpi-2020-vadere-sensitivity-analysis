import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import logging
import json

from pandas import DataFrame, Series
from datetime import datetime
from typing import Union

from SALib.util.results import ResultDict

from uq.utils.datatype import is_vector, unbox
from uq.utils.data_eval import sample_mean, sample_mode, sample_var, effective_sample_size
from uq.utils.general import get_current_uq_state, get_current_vadere_version, get_current_vadere_commit
from suqc.utils.general import get_current_suqc_state


class DataSaver:

    def __init__(self, in_path_to_tutorial: str = None, type: str = None):
        self.path_to_files, self.path_to_folder = self._create_output_dir(in_path_to_tutorial, type)

    def get_path_to_files(self):
        return self.path_to_files

    def set_path_to_files(self, in_path_to_files):
        self.path_to_files = in_path_to_files

    def write_to_file(self, variable, name: str, key: str, qoi: str):
        # write samples and results to file
        f = open(os.path.join(self.path_to_files, name + '.data'), 'w')
        f.write(repr(key) + "\t" + qoi + "\n")
        np.savetxt(f, np.transpose(variable))
        # f.write(repr(variable))
        f.close()

    def write_var_to_file(self, variable, name: str):
        f = open(os.path.join(self.path_to_files, name + '.data'), 'a+')
        if type(variable) == np.ndarray:
            while is_vector(variable) and np.ndim(variable) > 1:
                variable = unbox(variable)
            if np.ndim(variable) > 2:
                np.savetxt(f, variable.flatten())
            else:
                np.savetxt(f, variable)
        elif type(variable) == list:
            for item in variable:
                f.write("%s\n" % item)
        else:
            if type(variable) == ResultDict:
                for item in variable.to_df():
                    f.write("%s\n" % item)
            else:
                f.write("%s\n" % variable)
        f.close()

    def write_model_eval_to_file(self, key, qoi, parameters, results, bool_write: bool = True):
        if bool_write:
            filepath = os.path.join(self.path_to_files, 'model_evaluations.data')
            bool_first = not (os.path.isfile(filepath))

            f = open(filepath, 'a+')

            if bool_first:
                # write header
                f.write(repr(key) + "\t" + qoi + "\n")

            for i in range(0, len(parameters)):
                parameter_values = list(parameters[i].values())
                if type(results) == np.ndarray or type(results) == list:
                    qoi_values = np.array(results[i])
                else:
                    if type(results) == dict:
                        qoi_values = np.array([])
                        for key in results.keys():
                            tmp = np.array(results[key])
                            qoi_values = np.append(qoi_values, tmp)
                    else:
                        if type(results) == DataFrame:
                            qoi_values = results.loc[i]
                        else:
                            if type(results) == Series:
                                qoi_values = results.values[i]
                            else:
                                raise Warning("Datatype of results is not known / implemented.")

                f.write(' '.join(map(str, parameter_values)))
                f.write("\t")
                # todo: adapt for multi-dim QoI
                if np.ndim(qoi_values) == 0:
                    qoi_values = [qoi_values]
                f.write(' '.join(map(str, qoi_values)))
                f.write("\n")
            f.flush()
            f.close()

    def write_data_misfit_to_file(self, key, qoi, parameters, data_misfit):
        # write to readable file for post-processing with 3rd party software
        filepath = os.path.join(self.path_to_files, 'data_misfit.data')
        f = open(filepath, 'a+')
        # write header
        f.write(repr(key) + "\t Data misfit (" + qoi + ") \n")
        array_to_file = np.transpose(np.vstack((parameters, data_misfit)))
        f.write(' \n'.join(map(str, array_to_file)))
        f.flush()
        f.close()

        # write to pickle
        results = dict()
        results["parameters"] = parameters
        results["data_misfit"] = data_misfit
        results["key"] = key
        results["qoi"] = qoi

        with open(os.path.join(self.path_to_files, 'data_misfit.pickle'), 'wb') as file:
            pickle.dump(results, file)

    # private method
    @staticmethod
    def _create_output_dir(path2tutorial: str, type: str = None):
        if "results" not in path2tutorial:
            path_folder = os.path.abspath(os.path.join(path2tutorial, 'results'))
        else:
            path_folder = os.path.abspath(path2tutorial)
        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
        if type is not None:
            path_files = os.path.join(path_folder, date_str + "_" + type)
        else:
            path_files = os.path.join(path_folder, date_str)

        if not (os.path.isdir(path_files)):
            os.makedirs(path_files)

        return path_files, path_folder

    def save_figure(self, handle, name: str, path_to_files: str = None, bool_save_data: bool = True):

        if self.get_path_to_files() is None:
            if not path_to_files is None:
                self.set_path_to_files(path_to_files)

        if bool_save_data and not self.path_to_files is None:
            plt.savefig(os.path.join(self.get_path_to_files(), "fig_png_" + name + ".png"))
            plt.savefig(os.path.join(self.get_path_to_files(), "fig_pdf_" + name + ".pdf"))
            with open(os.path.join(self.get_path_to_files(), "fig_pickle_" + name + '_fig.pickle'),
                      'wb') as file:  # wb = write binary
                pickle.dump(handle, file)

            plt.close(handle)

    def save_to_file(self, path, data, name):
        if path is None and not self.get_path_to_files() is None:
            path = self.get_path_to_files()
        with open(os.path.join(path, name + ".pickle"), 'wb') as file:
            pickle.dump(data, file)

    def print_results(self, samples, burn_in, computation_time, acceptance_rate, nr_steps, dim):
        logger = logging.getLogger("Results")
        logger.info(' ')
        logger.info('Computation time:  \t\t\t\t\t\t\t{:.2f} min'.format(computation_time / 60), )
        logger.info('Mean of samples (without burn-in): \t\t\t{}'.format(sample_mean(samples, burn_in)))
        logger.info('Mode of samples (without burn-in): \t\t\t{}'.format(sample_mode(samples, burn_in)))
        logger.info('Var of samples (without burn-in): \t\t\t{}'.format(sample_var(samples, burn_in)))
        logger.info('Overall acceptance rate: \t\t\t\t\t{:.4f}'.format(acceptance_rate))
        # logger.info('Auto-correlation of samples (without burn_in): \t {}'.format(autocorrelation_samples(samples, burn_in)))
        ess = effective_sample_size(samples, burn_in, nr_steps, dim)
        logger.info('Effective sample size: \t\t\t\t\t\t{}'.format(ess))

    def write_results_to_file(self, folder, samples, burn_in, computation_time, acceptance_rate, nr_steps, dim,
                              jump_width):
        logger = logging.getLogger("utils.write_results_to_file")

        with open(os.path.join(folder, "results.txt"), 'w') as file:
            file.write(" *** Results *** \n")
            file.write('Computation time:  \t\t\t\t\t\t\t{} min\n'.format(computation_time / 60), )
            file.write('Mean of samples (without burn-in): \t\t\t{}\n'.format(sample_mean(samples, burn_in)))
            file.write('Mode of samples (without burn-in): \t\t\t{}\n'.format(sample_mode(samples, burn_in)))
            file.write('Var of samples (without burn-in): \t\t\t{}\n'.format(sample_var(samples, burn_in)))
            file.write('Overall acceptance rate: \t\t\t\t\t{}\n'.format(acceptance_rate))
            if jump_width is not None:
                file.write('Median of jump width: \t\t\t\t\t{}\n'.format(np.median(jump_width)))
            # logger.info('Auto-correlation of samples (without burn_in): \t {}'.format(autocorrelation_samples(samples, burn_in)))
            ess = effective_sample_size(samples, burn_in, nr_steps, dim)
            file.write('Effective sample size: \t\t\t\t\t\t{}\n'.format(ess))

            file.close()

    def write_sens_results_pce_to_file(self, total_sens, computation_time, M: int, order: int,
                                       computation_time_samples: float, key: Union[list, str], approx_model,
                                       no_evals: int = None):
        with open(os.path.join(self.get_path_to_files(), "results.txt"), 'a+') as file:
            file.write('** Parameters **: \n')
            file.write('Number of samples: \t\t\t {} \n'.format(M))
            if order is not None:
                file.write('Order of polynomial: \t\t {} \n'.format(order))
            file.write('Parameters investigated: \t {} \n'.format(key))
            if no_evals is not None:
                file.write('Number of model evaluations (samples): \t {} \n'.format(no_evals))

            file.write('** Results **: \n')
            if approx_model is not None:
                file.write('Approximation PC expansion: \t {} \n'.format(str(approx_model)))
            if computation_time_samples is not None:
                file.write('Computation time (samples): \t {} min \n'.format(computation_time_samples))

            file.write('Sensitivity indices: \t\t\t {} \n'.format(total_sens))
            file.write('Computation time (indices): \t {} min \n'.format(computation_time))

            file.write('\n')

            file.close()

    def write_sens_results_to_file(self, model: "Model", x_lower: np.ndarray, x_upper: np.ndarray,
                                   no_runs_averaged: int, n_samples: int, total_indices_constantine: np.ndarray,
                                   total_indices_jansen: np.ndarray, first_indices_jansen: np.ndarray,
                                   first_incides_saltelli_2010: np.ndarray, computation_time: float):

        self.write_var_to_file(total_indices_constantine, "total_indices_constantine")
        if first_indices_jansen is not None:
            self.write_var_to_file(total_indices_jansen, "total_indices_jansen")
            self.write_var_to_file(first_indices_jansen, "first_indices_jansen")
            self.write_var_to_file(first_incides_saltelli_2010, "first_incides_saltelli_2010")

        with open(os.path.join(self.get_path_to_files(), "results.txt"), 'a+') as file:
            file.write('** Parameters **: \n')
            file.write('Number of samples (N): \t\t\t\t\t {} \n'.format(n_samples))
            file.write("Number of runs averaged:  \t\t\t {} \n".format(no_runs_averaged))

            file.write('Parameters investigated: \t\t\t {} \n'.format(model.get_key()))
            file.write('Lower limits (parameters): \t\t\t {} \n'.format(x_lower))
            file.write('Upper limits (parameters): \t\t\t {} \n'.format(x_upper))

            file.write('Quantity of interest: \t\t\t\t {} \n'.format(model.get_qoi()))

            file.write('\n** Software versions **: \n')
            file.write('SUQ Controller Version (suqc): \t\t {} \n'.format(get_current_suqc_state()["suqc_version"]))
            file.write('UQ Software (git_commit_hash): \t\t {} \n'.format(get_current_uq_state()["git_hash"]))
            file.write('UQ Software (uncommited_changes):\t {} \n'.format(
                str.replace(get_current_uq_state()["uncommited_changes"], '\n M', ';')))

            file.write(
                'Vadere Software (version): \t\t\t {}'.format(get_current_vadere_version(model.get_path_2_model())))
            file.write(
                'Vadere Software (commit hash): \t\t {} \n'.format(get_current_vadere_commit(model.get_path_2_model())))
            file.write('Vadere Scenario: \t\t\t\t\t {} \n'.format(model.get_path_2_scenario()))

            file.write('\n** Results **: \n')

            file.write('Sobol total indices (Constantine): \t\t\t\t {} \n'.format(total_indices_constantine))
            file.write('Sobol total indices (Jansen): \t\t\t\t {} \n'.format(total_indices_jansen))

            file.write('Sobol first order indices (Jansen): \t\t\t {} \n'.format(first_indices_jansen))
            file.write('Sobol first order indices (Saltelli-2010): \t\t\t {} \n'.format(first_incides_saltelli_2010))

            file.write('Number of runs performed: \t\t\t {} \n'.format(model.get_n_evals()))

            file.write('\nComputation time: \t\t\t\t\t {} min \n'.format(computation_time))

            file.write('\n')
            file.close()

    def write_sens_results_salib_to_file(self, model: "Model", x_lower, x_upper, no_runs_averaged, n_samples,
                                         sobol_indices, computation_time):

        self.write_var_to_file(sobol_indices, "sobol_indices")

        if type(sobol_indices) == ResultDict:
            self.write_var_to_file(sobol_indices["ST"], "sobol_total_indices")

        with open(os.path.join(self.get_path_to_files(), "results.txt"), 'a+') as file:
            file.write('** Parameters **: \n')
            file.write('Number of samples (N): \t\t\t\t\t {} \n'.format(n_samples))
            file.write("Number of runs averaged:  \t\t\t {} \n".format(no_runs_averaged))

            file.write('Parameters investigated: \t\t\t {} \n'.format(model.get_key()))
            file.write('Lower limits (parameters): \t\t\t {} \n'.format(x_lower))
            file.write('Upper limits (parameters): \t\t\t {} \n'.format(x_upper))

            file.write('Quantity of interest: \t\t\t\t {} \n'.format(model.get_qoi()))

            file.write('\n** Software versions **: \n')
            file.write('SUQ Controller Version (suqc): \t\t {} \n'.format(get_current_suqc_state()["suqc_version"]))
            file.write('UQ Software (git_commit_hash): \t\t {} \n'.format(get_current_uq_state()["git_hash"]))
            file.write('UQ Software (uncommited_changes):\t {} \n'.format(
                str.replace(get_current_uq_state()["uncommited_changes"], '\n M', ';')))

            file.write(
                'Vadere Software (version): \t\t\t {}'.format(get_current_vadere_version(model.get_path_2_model())))
            file.write(
                'Vadere Software (commit hash): \t\t {} \n'.format(get_current_vadere_commit(model.get_path_2_model())))
            file.write('Vadere Scenario: \t\t\t\t\t {} \n'.format(model.get_path_2_scenario()))

            file.write('\n** Results **: \n')

            if type(sobol_indices) == ResultDict:
                file.write('Sobol total indices: \t\t\t\t {} \n'.format(sobol_indices["ST"]))
                file.write('Sobol first order indices: \t\t\t {} \n'.format(sobol_indices["S1"]))
            else:
                file.write('Sobol total indices: \t\t\t\t {} \n'.format(sobol_indices))

            file.write('Number of runs performed: \t\t\t {} \n'.format(model.get_n_evals()))

            file.write('\nComputation time: \t\t\t\t\t {} min \n'.format(computation_time))

            file.write('\n')
            file.close()

    def save_results_to_file(self, sampling_config, vadere_model):
        folder_results = self.get_path_to_files()
        sampling_config.dump_to_file(folder_results)
        vadere_model.dump_to_file(folder_results)

        return folder_results

    def save_parameters_to_file(self, key, qoi, path2tutorial, path2model, path2scenario, run_local, seed, nr_steps,
                                burn_in,
                                jump_width, prior_params, true_parameter_value, batch_jump_width,
                                acceptance_rate_limits,
                                bool_noise, bool_surrogate, nr_points_surrogate, limits_surrogate, dim):

        folder = self.get_path_to_files()

        tmp1 = "key = %s \nqoi = %s \npath2tutorial = %s \npath2model = %s \npath2scenario = %s \nrun_local = %s\n" % (
            key, qoi, path2tutorial, path2model, path2scenario, run_local)
        if not type(true_parameter_value) is np.ndarray:
            true_parameter_value = np.array(true_parameter_value)

        tmp2 = "seed = %d\nnr_steps = %d\nburn_in = %d\njump_width = %f\nprior_params = %s\ntrue_parameter_value = %s\n" \
               % (seed, nr_steps, burn_in, jump_width, prior_params, np.array2string(true_parameter_value))
        tmp3 = "batch_jump_width = %f \nacceptance_rate_limits = [%f,%f] \nbool_noise =%s" % (
            batch_jump_width, acceptance_rate_limits[0], acceptance_rate_limits[1], bool_noise)
        if bool_surrogate:
            if dim == 1:
                tmp4 = "\n\nSurrogate parameters:\nnr_points_surrogate = %d \n, limits_surrogate = [%f, %f]\n" % \
                       (nr_points_surrogate, limits_surrogate[0], limits_surrogate[1])
            else:
                if dim == 1:
                    lower_limit = np.array(limits_surrogate[0])
                    upper_limit = np.array(limits_surrogate[1])
                else:
                    lower_limit = limits_surrogate[:, 0]
                    upper_limit = limits_surrogate[:, 1]

                tmp4 = "\n\nSurrogate parameters:\nnr_points_surrogate = %d \n, limits_surrogate = [%s, %s]\n" % \
                       (nr_points_surrogate, np.array2string(lower_limit), np.array2string(upper_limit))
        else:
            tmp4 = ""

        if bool_surrogate:
            filename = "parameters_surrogate.txt"

        else:
            filename = "parameters_vadere_eval.txt"

        with open(os.path.join(folder, filename), 'w') as file:
            file.write(tmp1 + tmp2 + tmp3 + tmp4)
            file.close()
