# third party imports
import unittest
import numpy as np
from math import log, pi

# local imports
from probeye.noise import NoiseTemplate
from probeye.noise import NormalNoiseZeroMean


class TestProblem(unittest.TestCase):

    def test_noise_template(self):
        # check normal use cases
        noise_template = NoiseTemplate(['bias', 'sigma'])
        self.assertEqual(noise_template.prms_def,
                         {'bias': 'bias', 'sigma': 'sigma'})
        self.assertEqual(noise_template.prms_dim, 2)
        noise_template = NoiseTemplate(['b', {'sd': 'sigma'}])
        self.assertEqual(noise_template.prms_def,
                         {'b': 'b', 'sd': 'sigma'})
        self.assertEqual(noise_template.prms_dim, 2)
        # check invalid initialization
        with self.assertRaises(TypeError):
            # noinspection PyTypeChecker
            NoiseTemplate('sigma')

    def test_normal_noise_zero_mean(self):
        # check valid initialization
        noise_model = NormalNoiseZeroMean(['sigma'])
        self.assertEqual(noise_model.sigma_name, 'sigma')
        noise_model = NormalNoiseZeroMean(['weird_name'])
        self.assertEqual(noise_model.sigma_name, 'weird_name')
        # check invalid initialization (provided more than one parameter)
        with self.assertRaises(RuntimeError):
            NormalNoiseZeroMean(['bias', 'sd'])
        # check the log-likelihood-contribution method
        noise_model = NormalNoiseZeroMean(['sigma'])
        # one error of zero
        sigma = 2.1
        error = np.array([0.0])
        prms = {'sigma': sigma}
        expected_result = -log(sigma) - 1 / 2 * log(2 * pi)
        self.assertEqual(noise_model.loglike_contribution(error, prms),
                         expected_result)
        # two errors of zero
        sigma = 2.1
        error = np.array([0.0, 0.0])
        prms = {'sigma': sigma}
        expected_result = 2 * (-log(sigma) - 1 / 2 * log(2 * pi))
        self.assertEqual(noise_model.loglike_contribution(error, prms),
                         expected_result)
        # one non-zero error
        sigma = 2.1
        error = np.array([1.1])
        prms = {'sigma': sigma}
        expected_result = -log(sigma) - 1 / 2 * log(2 * pi) \
                          - 1 / 2 * np.sum(np.square(error / sigma))
        self.assertEqual(noise_model.loglike_contribution(error, prms),
                         expected_result)
        # two non-zero errors
        sigma = 2.1
        error = np.array([1.1, -0.5])
        prms = {'sigma': sigma}
        expected_result = 2 * (-log(sigma) - 1 / 2 * log(2 * pi)) \
                          - 1 / 2 * np.sum(np.square(error / sigma))
        self.assertEqual(noise_model.loglike_contribution(error, prms),
                         expected_result)

if __name__ == "__main__":
    unittest.main()
