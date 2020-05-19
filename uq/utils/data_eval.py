import numpy as np
from scipy import stats

from uq.utils.datatype import unbox
from statsmodels.graphics.tsaplots import acf

""" ----------------------------------- sample mean / var  --------------------------------------------------- """


def sample_mean(samples: np.ndarray, burn_in: int = 0):
    return np.mean(samples[burn_in:len(samples)])


def sample_mode(samples: np.ndarray, burn_in: int = 0):
    # most common value (for asymmetric distribution)
    mode, count = stats.mode(samples[burn_in:len(samples)])
    # stats.mode is only for discrete data
    count, bins = np.histogram(samples[burn_in:len(samples)],
                               np.max([int(len(samples[burn_in:len(samples)]) / 100), 10]))
    idx = np.argmax(count)
    mode = (bins[idx] + bins[idx + 1]) / 2
    return unbox(mode)


def sample_std(samples: np.ndarray, burn_in: int = 0):
    return np.std(samples[burn_in:len(samples)])


def sample_var(samples: np.ndarray, burn_in: int = 0):
    return np.var(samples[burn_in:len(samples)])


""" ----------------------------------- averaging of samples --------------------------------------------------- """


def averaging_simulation_data(x_data: np.ndarray, y_data: np.ndarray, nr_points_averaged: int):
    x_reshaped = np.reshape(x_data, newshape=(-1, nr_points_averaged))
    y_reshaped = np.reshape(y_data, newshape=(-1, nr_points_averaged))
    y_av = np.median(y_reshaped, axis=1)
    x_av = np.median(x_reshaped, axis=1)
    return x_av, y_av


""" ----------------------------------- effective samples size --------------------------------------------------- """


def calc_acf_samples(samples: np.ndarray, burn_in: int, dim: int):
    if samples is not None:
        if dim == 1 and len(samples) > 1:
            n_samples = len(samples)
            samples_wo_burn_in = samples[burn_in:n_samples]
            acf_values = acf(samples_wo_burn_in, fft=True, nlags=n_samples - burn_in)

        else:
            # n_samples = np.size(samples, axis=1)
            # samples_wo_burn_in = samples[:, burn_in:n_samples]
            # todo: acf for multi-dimensional input
            acf_values = None
            raise Warning('utils.calc_acf_samples not defined for multi-dimensional input')

    return acf_values


def effective_sample_size(samples: np.ndarray, burn_in: int, nr_steps: int, dim: int):
    if dim == 1 and len(samples) > 1:
        # from kruschke-2015, p. 184
        acf_samples = calc_acf_samples(samples, burn_in, dim)
        # % ACF
        # % acf = autocorr(tau_save, min(n_steps - 1, 200));
        idx_acf = np.argmax(acf_samples <= 0.05)

        if idx_acf > 0:  # entry was found
            effective_size = (nr_steps - burn_in) / (1 + 2 * np.sum(acf_samples[1:idx_acf]))
        else:
            effective_size = 0

    else:
        effective_size = None
        # todo: effective sample size for multi-dimensional parameter space (how to calculcate acf?)

    return effective_size


def autocorrelation_samples(samples: np.ndarray, burn_in: int):
    # from https://ipython-books.github.io/103-computing-the-autocorrelation-of-a-time-series/
    result = np.correlate(samples[burn_in:len(samples)], samples[burn_in:len(samples)], mode='full')
    return result[result.size // 2:]  # normalized
