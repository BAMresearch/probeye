# standard library
from math import log, pi

# third party imports
import unittest
import numpy as np

# local imports
from probeye.inference.taralli_.noise_models import NormalNoise


class TestProblem(unittest.TestCase):

    def test_normal_noise_init(self):
        # check valid initialization
        noise_model = NormalNoise(['sigma'], 'y')
        self.assertEqual(noise_model.sigma_name, 'sigma')
        noise_model = NormalNoise(['weird_name'], 'y')
        self.assertEqual(noise_model.sigma_name, 'weird_name')
        self.assertEqual(noise_model.noise_type, 'additive')
        noise_model = NormalNoise(['sigma'], 'y', noise_type='multiplicative')
        self.assertEqual(noise_model.noise_type, 'multiplicative')
        # check invalid initialization
        with self.assertRaises(RuntimeError):
            # more than one noise model parameter given (only one valid here)
            NormalNoise(['bias', 'sd'], 'y')
        with self.assertRaises(ValueError):
            # invalid noise_type given
            NormalNoise(['sigma'], 'y', noise_type='invalid_type')

    def test_normal_noise_error_function(self):
        # simply check that the error_function is set correctly when the noise
        # model is initialized
        ym_dict, ye_dict = {'y': 4}, {'y': 2}
        noise_model = NormalNoise(['sigma'], 'y', noise_type='additive')
        computed_value = noise_model.error_function(ym_dict, ye_dict)
        expected_value = {'y': 2}
        self.assertEqual(computed_value, expected_value)
        noise_model = NormalNoise(['sigma'], 'y', noise_type='multiplicative')
        computed_value = noise_model.error_function(ym_dict, ye_dict)
        expected_value = {'y': 1.0}
        self.assertEqual(computed_value, expected_value)

    def test_normal_noise_error_function_additive(self):
        # check method for scalar values
        ym_dict, ye_dict = {'y': 4}, {'y': 2}
        noise_model = NormalNoise(['sigma'], 'y')
        computed_value = noise_model.error_function_additive(ym_dict, ye_dict)
        expected_value = {'y': 2.0}
        self.assertEqual(computed_value, expected_value)
        # check method for numpy-arrays
        ym_dict, ye_dict = {'y': np.array([4, 5])}, {'y': np.array([2, 4])}
        noise_model = NormalNoise(['sigma'], 'y')
        computed_value = noise_model.error_function_additive(ym_dict, ye_dict)
        expected_value = {'y': np.array([2, 1])}
        self.assertTrue(np.allclose(computed_value['y'], expected_value['y']))

    def test_normal_noise_error_function_multiplicative(self):
        # check method for scalar values
        ym_dict, ye_dict = {'y': 4}, {'y': 2}
        noise_model = NormalNoise(['sigma'], 'y')
        comp_value = noise_model.error_function_multiplicative(ym_dict, ye_dict)
        expected_value = {'y': 1.0}
        self.assertEqual(comp_value, expected_value)
        # check method for numpy-arrays
        ym_dict, ye_dict = {'y': np.array([4, 5])}, {'y': np.array([2, 4])}
        noise_model = NormalNoise(['sigma'], 'y')
        comp_value = noise_model.error_function_multiplicative(ym_dict, ye_dict)
        expected_value = {'y': np.array([1.0, 0.25])}
        self.assertTrue(np.allclose(comp_value['y'], expected_value['y']))

    def test_normal_noise_loglike_contribution(self):
        # prepare the first setup for the tests (here all add. errors are zero)
        noise_model = NormalNoise('sigma', 'y')
        model_response_dict = {'Exp1': {'x': 1, 'y': 3},
                               'Exp2': {'x': 3, 'y': 6},
                               'Exp3': {'x': 5, 'y': 9}}
        problem_experiments = {'Exp1': {'sensor_values': {'x': 1, 'y': 3},
                                        'forward_model': 'TestModel'},
                               'Exp2': {'sensor_values': {'x': 3, 'y': 6},
                                        'forward_model': 'TestModel'},
                               'Exp3': {'sensor_values': {'x': 5, 'y': 9},
                                        'forward_model': 'TestModel'}}
        # the following is usually done automatically when adding the noise
        # model to the inference problem
        noise_model.problem_experiments = problem_experiments
        noise_model.add_experiment_names(['Exp1', 'Exp2', 'Exp3'])
        sigma = 2.1
        prms = {'sigma': sigma}
        computed_result = noise_model.loglike_contribution(
            model_response_dict, prms)
        expected_result = 3 * (-log(sigma) - 1 / 2 * log(2 * pi))
        self.assertAlmostEqual(computed_result, expected_result)

        # prepare the second setup for the tests (add. errors are non-zero)
        noise_model = NormalNoise('sigma', 'y')
        model_response_dict = {'Exp1': {'x': 1, 'y': 3},
                               'Exp2': {'x': 3, 'y': 6},
                               'Exp3': {'x': 5, 'y': 9}}
        problem_experiments = {'Exp1': {'sensor_values': {'x': 1, 'y': 4},
                                        'forward_model': 'TestModel'},
                               'Exp2': {'sensor_values': {'x': 3, 'y': 8},
                                        'forward_model': 'TestModel'},
                               'Exp3': {'sensor_values': {'x': 5, 'y': 7},
                                        'forward_model': 'TestModel'}}
        # the following is usually done automatically when adding the noise
        # model to the inference problem
        noise_model.problem_experiments = problem_experiments
        noise_model.add_experiment_names(['Exp1', 'Exp2', 'Exp3'])
        sigma = 1.1
        prms = {'sigma': sigma}
        computed_result = noise_model.loglike_contribution(
            model_response_dict, prms)
        expected_error = {'y': np.array([-1, -2, 2])}
        expected_result = 3 * (-log(sigma) - 1 / 2 * log(2 * pi)) - 1 / 2 \
                          * np.sum(np.square(expected_error['y'] / sigma))
        self.assertAlmostEqual(computed_result, expected_result)

if __name__ == "__main__":
    unittest.main()
