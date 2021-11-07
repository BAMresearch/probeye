# standard library imports
import unittest

# third party imports
import numpy as np

# local imports
from probeye.definition.noise_model import NoiseModelBase
from probeye.definition.noise_model import NormalNoiseModel
from probeye.definition.sensor import Sensor


class TestProblem(unittest.TestCase):

    def test_init(self):
        # initialize an instance and check if everything is there
        noise_template = NoiseModelBase(
            'normal', ['bias', 'sigma'], Sensor('y'))
        self.assertEqual(noise_template.prms_def,
                         {'bias': 'bias', 'sigma': 'sigma'})
        self.assertEqual(noise_template.prms_dim, 2)
        noise_template = NoiseModelBase(
            'normal', ['b', {'sd': 'sigma'}], Sensor('y'))
        self.assertEqual(noise_template.prms_def,
                         {'b': 'b', 'sd': 'sigma'})
        self.assertEqual(noise_template.prms_dim, 2)
        self.assertEqual(noise_template.sensor_names, ['y'])
        self.assertEqual(noise_template.experiment_names, [])

    def test_add_experiment_names(self):
        # prepare the setup for the tests
        noise_template = NoiseModelBase(
            'normal', ['bias', 'sigma'], Sensor('y'))
        problem_experiments = {'Exp1': {'sensor_values': {'x': 1, 'y': 2},
                                        'forward_model': 'TestModel'},
                               'Exp2': {'sensor_values': {'x': 3, 'y': 4},
                                        'forward_model': 'TestModel'},
                               'Exp3': {'sensor_values': {'x': 5, 'y': 6},
                                        'forward_model': 'TestModel'},
                               'Exp4': {'sensor_values': {'x': 7, 'z': 8},
                                        'forward_model': 'TestModel'},
                               'Exp5': {'sensor_values': {'x': 7, 'y': 8},
                                        'forward_model': 'TestModel'},
                               'Exp6': {'sensor_values': {'x': 7, 'y': 8},
                                        'forward_model': 'OtherModel'}}
        # the following is usually done automatically when adding the noise
        # model to the inference problem
        noise_template.problem_experiments = problem_experiments
        # perform some simple tests for valid usage
        noise_template.add_experiments('Exp1')
        self.assertEqual(noise_template.experiment_names, ['Exp1'])
        noise_template.add_experiments(['Exp2', 'Exp3'])
        self.assertEqual(noise_template.experiment_names,
                         ['Exp1', 'Exp2', 'Exp3'])
        # now check some invalid input arguments
        with self.assertRaises(RuntimeError):
            # add an experiment, that does not have the noise model's sensors
            noise_template.add_experiments('Exp4')
        with self.assertRaises(RuntimeError):
            # add experiments that refer to more than one forward model
            noise_template.add_experiments(['Exp5', 'Exp6'])
        with self.assertRaises(RuntimeError):
            # adding the same experiment again
            noise_template.add_experiments('Exp1')

    def test_error(self):
        # prepare the setup for the tests
        noise_template = NoiseModelBase('normal', ['bias', 'sigma'],
                                        [Sensor('y1'), Sensor('y2')])
        # in the NoiseModelBase class the error_function is just a dummy,
        # so we need to define it to be able to test the error-method
        noise_template.error_function =\
            lambda d1, d2: {name: d1[name] - d2[name]
                            for name in noise_template.sensor_names}
        model_response_dict = {'Exp1': {'x': 1, 'y1': 3, 'y2': -3},
                               'Exp2': {'x': 3, 'y1': 6, 'y2': -6},
                               'Exp3': {'x': 5, 'y1': 9, 'y2': -9}}
        problem_experiments = {'Exp1': {'sensor_values': {'x': 1,
                                                          'y1': 2,
                                                          'y2': -2},
                                        'forward_model': 'TestModel'},
                               'Exp2': {'sensor_values': {'x': 3,
                                                          'y1': 4,
                                                          'y2': -4},
                                        'forward_model': 'TestModel'},
                               'Exp3': {'sensor_values': {'x': 5,
                                                          'y1': 6,
                                                          'y2': -6},
                                        'forward_model': 'TestModel'}}
        # the following is usually done automatically when adding the noise
        # model to the inference problem
        noise_template.problem_experiments = problem_experiments
        noise_template.add_experiments(['Exp1', 'Exp2', 'Exp3'])
        # now we can call the error-method
        computed_value = noise_template.error(model_response_dict)
        expected_value = {'y1': np.array([1., 2., 3.]),
                          'y2': np.array([-1., -2., -3.])}
        # due to the numpy arrays we have to loop over everything to compare
        for sensor_name, array in computed_value.items():
            for i in range(len(array)):
                self.assertEqual(array[i], expected_value[sensor_name][i])

    def test_normal_noise_init(self):
        # check valid initialization
        noise_model = NormalNoiseModel({'sigma': 'std'}, Sensor('y'))
        self.assertTrue(noise_model.zero_mean)
        self.assertEqual(noise_model.noise_type, 'additive')
        noise_model = NormalNoiseModel({'sigma': 'std'}, Sensor('y'),
                                       noise_type='multiplicative')
        self.assertEqual(noise_model.noise_type, 'multiplicative')
        noise_model = NormalNoiseModel({'sigma': 'std'}, Sensor('y'),
                                       noise_type='other')
        self.assertEqual(noise_model.noise_type, 'other')
        noise_model = NormalNoiseModel({'mean': 'mean', 'sigma': 'std'},
                                       Sensor('y'))
        self.assertTrue(not noise_model.zero_mean)
        # check invalid initialization
        with self.assertRaises(ValueError):
            # invalid noise_type given
            NormalNoiseModel(
                {'sigma': 'std'}, Sensor('y'), noise_type='invalid_type')
        with self.assertRaises(RuntimeError):
            # no 'std'-parameter given
            NormalNoiseModel({'sigma': "this should be 'std'"}, Sensor('y'))

    def test_normal_noise_error_function(self):
        # simply check that the error_function is set correctly when the noise
        # model is initialized
        ym_dict, ye_dict = {'y': 4}, {'y': 2}
        noise_model = NormalNoiseModel({'sigma': 'std'}, Sensor('y'),
                                       noise_type='additive')
        computed_value = noise_model.error_function(ym_dict, ye_dict)
        expected_value = {'y': 2}
        self.assertEqual(computed_value, expected_value)
        noise_model = NormalNoiseModel({'sigma': 'std'}, Sensor('y'),
                                       noise_type='multiplicative')
        computed_value = noise_model.error_function(ym_dict, ye_dict)
        expected_value = {'y': 1.0}
        self.assertEqual(computed_value, expected_value)
        # the 'other' error function is not pre-defined
        noise_model = NormalNoiseModel({'sigma': 'std'}, Sensor('y'),
                                       noise_type='other')
        with self.assertRaises(NotImplementedError):
            noise_model.error_function(ym_dict, ye_dict)

    def test_normal_noise_error_function_additive(self):
        # check method for scalar values
        ym_dict, ye_dict = {'y': 4}, {'y': 2}
        noise_model = NormalNoiseModel({'sigma': 'std'}, Sensor('y'))
        computed_value = noise_model.error_function_additive(ym_dict, ye_dict)
        expected_value = {'y': 2.0}
        self.assertEqual(computed_value, expected_value)
        # check method for numpy-arrays
        ym_dict, ye_dict = {'y': np.array([4, 5])}, {'y': np.array([2, 4])}
        noise_model = NormalNoiseModel({'sigma': 'std'}, Sensor('y'))
        computed_value = noise_model.error_function_additive(ym_dict, ye_dict)
        expected_value = {'y': np.array([2, 1])}
        self.assertTrue(np.allclose(computed_value['y'], expected_value['y']))

    def test_normal_noise_error_function_multiplicative(self):
        # check method for scalar values
        ym_dict, ye_dict = {'y': 4}, {'y': 2}
        noise_model = NormalNoiseModel({'sigma': 'std'}, Sensor('y'))
        comp_value = noise_model.error_function_multiplicative(ym_dict, ye_dict)
        expected_value = {'y': 1.0}
        self.assertEqual(comp_value, expected_value)
        # check method for numpy-arrays
        ym_dict, ye_dict = {'y': np.array([4, 5])}, {'y': np.array([2, 4])}
        noise_model = NormalNoiseModel({'sigma': 'std'}, Sensor('y'))
        comp_value = noise_model.error_function_multiplicative(ym_dict, ye_dict)
        expected_value = {'y': np.array([1.0, 0.25])}
        self.assertTrue(np.allclose(comp_value['y'], expected_value['y']))

    def test_normal_noise_loglike_contribution(self):
        # the loglike-contribution is not defined in this general class
        noise_model = NormalNoiseModel({'sigma': 'std'}, Sensor('y'))
        with self.assertRaises(NotImplementedError):
            noise_model.loglike_contribution({}, None)

if __name__ == "__main__":
    unittest.main()
