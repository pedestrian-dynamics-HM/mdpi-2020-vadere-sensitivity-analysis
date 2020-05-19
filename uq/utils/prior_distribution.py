import scipy.stats as scipy_stats
import numpy as np
from typing import Union
from numpy.random import RandomState
import warnings

from uq.utils.datatype import box, is_scalar, nr_entries, unbox


class Prior(object):

    def __init__(self, in_type: str = None, in_params: dict = None, in_limits: np.ndarray = None):
        if in_type is None:
            self.type = in_params["Type"]

        self.type = in_type
        self.params = in_params
        self.limits = in_limits
        self.distribution = self.translate_inputs_to_distribution(self.type, self.params, self.limits)

    def get_distribution(self):
        return self.distribution

    def get_mean(self) -> float:
        return self.get_distribution().get_mean()

    # todo: replace by factory

    @staticmethod
    def translate_inputs_to_distribution(in_type: Union[str, dict], in_params: None, in_limits: np.ndarray):
        if type(in_type) is str:  # one-dim
            if in_type is 'Normal':
                if in_limits is not None:
                    distribution = GaussianGenTrunc(in_params["Mean"], in_params["Variance"], in_limits)
                else:
                    distribution = GaussianGen(in_params["Mean"], in_params["Variance"])
            elif in_type == 'Uniform':
                distribution = UniformGen(in_params["Low"], in_params["High"])
            else:
                raise Exception('Onedimensional - this type is not supported')
        elif type(in_type) is dict:  # multi-dim
            if in_type["Type"] == 'Uniform':
                distribution = UniformGenMult(in_type["Low"], in_type["High"], np.size(in_type["Low"]))
            elif in_type["Type"] == 'Normal':
                if np.size(in_type["Mean"]) == 1:
                    if in_limits is not None:
                        distribution = GaussianGenTrunc(in_type["Mean"], in_type["Variance"], in_limits)
                    else:
                        distribution = GaussianGen(in_type["Mean"], in_type["Variance"])
                else:
                    distribution = GaussianGenMult(in_type["Mean"], in_type["Variance"], np.size(in_type["Mean"]))
            else:
                raise Exception('Multidimensional - this type is not supported')

        return distribution

    def eval_prior(self, value) -> float:
        return self.get_distribution().get_pdf(value)


class GaussianGen(Prior):

    def __init__(self, mean: float = 0.0, deviation: float = 1.0):
        self.mean = mean
        self.deviation = deviation
        # super(gaussian_gen,self).__init__()

    def get_mean(self) -> float:
        return self.mean

    def get_deviation(self) -> float:
        return self.deviation

    def get_pdf(self, x) -> float:
        return scipy_stats.norm.pdf(x, self.get_mean(), self.get_deviation())

    def _pdf(self, x) -> float:
        # return np.exp(-(x - self.mean) ** 2 / (2. * self.variance)) / np.sqrt(2.0 * np.pi)
        warnings.warn("Accessing GaussianGen._pdf()")
        return scipy_stats.norm.pdf(x, self.get_mean(), self.get_deviation())

    def sample(self, n: int, random_state: RandomState = None) -> np.ndarray:
        if random_state is None:
            warnings.warn(
                "GaussianGen.sample() is called without RandomState object. Consider adding a RandomState object.")
        samples = scipy_stats.norm.rvs(loc=self.get_mean(), scale=self.get_deviation(), size=n,
                                       random_state=random_state)
        return samples


class GaussianGenTrunc(Prior):

    def __init__(self, mean: float, deviation: float, limits: np.ndarray):
        self.mean = mean
        self.deviation = deviation
        self.limits = limits

        a = (limits[0] - mean) / deviation
        b = (limits[1] - mean) / deviation
        self.a = a
        self.b = b

    def get_lower_limit(self) -> float:
        return self.limits[0]

    def get_a(self) -> float:
        return self.a

    def get_b(self) -> float:
        return self.b

    def get_upper_limit(self) -> float:
        return self.limits[1]

    def get_mean(self) -> float:
        return self.mean

    def get_deviation(self) -> float:
        return self.deviation

    def _pdf(self, x) -> float:
        warnings.warn("GaussianGenTrunc._pdf() is accessed instead of get_pdf()")
        pdf = scipy_stats.truncnorm(x, self.a, self.b)
        return pdf

    def get_pdf(self, x) -> float:
        pdf = scipy_stats.truncnorm(x, self.a, self.b)
        return pdf

    def sample(self, n: int, random_state: RandomState = None) -> np.ndarray:
        if random_state is None:
            warnings.warn("GaussianGenTrunc.sample() is called without a RandomState object!")
        samples = scipy_stats.truncnorm.rvs(self.get_a(), self.get_b(), size=n,
                                            random_state=random_state)  # standard-normal samples
        scaled_samples = samples * self.get_deviation() + self.get_mean()

        assert (np.all(scaled_samples >= self.get_lower_limit()))
        assert (np.all(scaled_samples <= self.get_upper_limit()))

        return scaled_samples


class UniformGen(Prior):
    """Uniform distribution"""

    def __init__(self, lower: float = -1, upper: float = 1):
        self.lower = lower
        self.upper = upper

    def get_upper(self) -> float:
        return self.upper

    def get_lower(self) -> float:
        return self.lower

    def get_mean(self) -> float:
        return np.mean([self.get_lower(), self.get_upper()])

    def sample(self, n: int, random_state: RandomState = None) -> np.ndarray:
        if random_state is None:
            warnings.warn("UniformGen.sample(): random_state is None. Consider passing a RandomState element.")
        samples = scipy_stats.uniform.rvs(loc=self.get_lower(), scale=self.get_upper() - self.get_lower(), size=n,
                                          random_state=random_state)
        # np.random.rand(n)*(self.get_upper()-self.get_lower()) + self.get_lower()
        return samples

    def _pdf(self, x) -> np.ndarray:
        warnings.warn("UniformGen._pdf() is accessed instead of get_pdf")
        return scipy_stats.uniform.pdf(x, loc=self.get_lower(), scale=self.get_upper() - self.get_lower())

    def get_pdf(self, x) -> np.ndarray:
        pdf = scipy_stats.uniform.pdf(x, loc=self.get_lower(), scale=self.get_upper() - self.get_lower())
        return pdf

class UniformGenMult(Prior):
    """Multivariate uniform distribution (non-correlated variables) """

    def __init__(self, lower: np.ndarray, upper: np.ndarray, dim: int):
        self.lower = lower
        self.upper = upper
        self.dim = dim

    def get_dim(self) -> int:
        return self.dim

    def get_lower(self) -> np.ndarray:
        return self.lower

    def get_upper(self) -> np.ndarray:
        return self.upper

    def get_dim(self) -> int:
        return self.dim

    def get_mean(self) -> np.ndarray:
        # todo: make sure this has the right dimension
        tmp = np.mean([self.get_lower(), self.get_upper()], axis=1)
        return tmp

    def get_pdf(self, x: np.ndarray) -> np.ndarray:
        """ todo: make sure that this is correct"""
        if self.get_dim() > 1:
            if np.ndim(x) == 1:
                x = box(x)
            m = np.size(x, axis=0)
            n = np.size(x, axis=1)
            if m == n:
                nr_values = m
            elif m == self.get_dim():
                nr_values = n
            elif n == self.get_dim():
                nr_values = m
                x = np.transpose(
                    x)  # first dimension is supposed to be the dimension of x, [:,i] should give all realizations for one dimension
            else:
                raise Exception("In one dimension, x needs to be equals to the dimension of the distribution")
        else:
            nr_values = nr_entries(x)
            while np.ndim(x) < 2:
                x = box(x)

        if is_scalar(x):
            x = box(x)

        pdf_val = np.zeros(shape=(self.get_dim(), nr_values))
        for i in range(0, self.get_dim()):
            dist = scipy_stats.uniform(loc=box(self.get_lower())[i],
                                       scale=box(self.get_upper())[i] - box(self.get_lower())[i])
            if nr_values == 1:
                pdf_val[i] = unbox(dist.pdf(x[i]))
            else:
                pdf_val[i] = unbox(dist.pdf(x[:, i]))

            # pdf_val[i] = scipy_stats.uniform.pdf(box(x)[i], loc=box(self.get_lower())[i], \
            # scale=box(self.get_upper())[i] - box(self.get_lower())[i])

        # return scipy_stats.uniform.pdf(x, loc=self.lower, scale=self.upper-self.lower)

        if self.get_dim() > 1:
            # todo: test
            joint_pdf_val = np.prod(pdf_val)
        else:
            joint_pdf_val = pdf_val

        assert joint_pdf_val.all() >= 0  # and np.prod(pdf_val) <= 1

        return joint_pdf_val

    def sample(self, n: int, random_state: RandomState = None) -> np.ndarray:
        if self.get_dim() == 1:
            sample_vec = UniformGen(lower=self.get_lower(), upper=self.get_upper()).sample(n)
        else:
            sample_vec = np.zeros(shape=(self.get_dim(), n))
            for i in range(0, self.get_dim()):
                sample_vec[i, :] = UniformGen(lower=self.get_lower()[i], upper=self.get_upper()[i]).sample(n,
                                                                                                           random_state=random_state)

        return sample_vec

    def eval_prior(self, value: np.ndarray):
        return super().eval_prior(self, value)


class GaussianGenMult(Prior):
    """ Multivariate Normal Distribution (non-correlated variables)"""

    def __init__(self, mean: np.ndarray, cov: np.ndarray, dim: int):
        self.mean = mean
        self.cov = cov
        self.dim = dim

    def get_mean(self) -> np.ndarray:
        return self.mean

    def get_dim(self) -> int:
        return self.dim

    def get_cov(self) -> np.ndarray:
        return self.cov

    def sample(self, n: int) -> np.ndarray:
        """ Only works for non-correlated variables"""
        sample_vec = np.zeros(shape=(self.get_dim(), n))
        for i in range(0, self.get_dim()):
            sample_vec[i, :] = GaussianGen(mean=self.get_mean()[i], deviation=self.get_cov()[i]).sample(n)
        return sample_vec

    def _pdf(self, x: np.ndarray) -> float:
        warnings.warn("GaussianGenMult._pdf() is accesssed instead of get_pdf()")
        tmp = scipy_stats.multivariate_normal.pdf(x, mean=self.get_mean(), cov=self.get_cov())
        # tmp should be one-dimensional
        return tmp

    def get_pdf(self, x: np.ndarray) -> float:
        tmp = scipy_stats.multivariate_normal.pdf(x, mean=self.get_mean(), cov=self.get_cov())
        # tmp should be one-dimensional
        return tmp


if __name__ == "__main__":
    params = {"mean": 0.0, "variance": 1.0}
    print(params)
    my_prior = Prior("Normal", params)
    print(my_prior.eval_prior(value=3.0))

    my_prior_uniform = Prior("Uniform", {"low": -1, "high": 1})
    print(my_prior_uniform.eval_prior(value=1.))
    print(my_prior_uniform.eval_prior(value=2.))

    print(my_prior_uniform.eval_prior(value=np.array([1, 2])))

    print()
