import unittest
from uq.utils.prior_distribution import UniformGenMult, UniformGen, GaussianGen
import numpy as np
from numpy.random import RandomState
import numpy.testing as nptest
import scipy
from scipy import stats


class MyTestCase(unittest.TestCase):

    def test_mean_uniform_gen(self):
        prior = UniformGen(-1, 1)
        self.assertEqual(prior.get_mean(), 0)

        prior = UniformGen(-10, 5)
        self.assertEqual(prior.get_mean(), -2.5)

    def test_mean_uniform_gen_mult(self):
        prior = UniformGenMult([-3, -3], [5, 5], 2)
        nptest.assert_allclose(prior.get_mean(), [-3, 5])

        prior = UniformGenMult([-3, 3], [5, -5], 2)
        nptest.assert_allclose(prior.get_mean(), [0, 0])

    def test_mean_gaussian_gen(self):
        prior = GaussianGen()
        self.assertEqual(prior.get_mean(), 0)

        prior = GaussianGen(-3, 0.1236)
        self.assertEqual(prior.get_mean(), -3)

    def test_sampling_uniform(self):
        seed = (np.abs(np.random.rand(1)*(2**30)).astype(int))
        dim = int(6)
        n_samples = int(1e5)
        x_lower = -1 * np.ones(shape=(dim, 1))
        x_upper = np.ones(shape=(dim, 1))
        prior = UniformGenMult(x_lower, x_upper, dim)
        samples = prior.sample(n_samples, RandomState(seed))
        #  for i in range(0, dim):
        #      plt.figure()
        #      plt.hist(samples[i, :])

        # make sure the largest / smallest samples are close to the bounds
        nptest.assert_allclose(np.max(samples, axis=1), x_upper.flatten(), rtol=1e-4)
        nptest.assert_allclose(np.min(samples, axis=1), x_lower.flatten(), rtol=1e-4)

        # assure that the samples are within the bounds
        self.assertTrue((samples <= x_upper).all())
        self.assertTrue((samples >= x_lower).all())

        # test distribution
        alpha = 0.05
        for i in range(0, dim):
            [D, pvalue] = scipy.stats.kstest(samples[i, :], stats.uniform(loc=x_lower, scale=(x_upper - x_lower)).cdf)
            self.assertGreater(pvalue, alpha)


if __name__ == '__main__':
    unittest.main()
