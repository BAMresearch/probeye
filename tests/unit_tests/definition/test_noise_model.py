# third party imports
import unittest
import numpy as np

# local imports
from probeye.definition.noise_model import NoiseModelTemplate


class TestProblem(unittest.TestCase):

    def test_init(self):
        # initialize an instance and check if everything is there
        noise_template = NoiseModelTemplate(['bias', 'sigma'], 'y')
        self.assertEqual(noise_template.prms_def,
                         {'bias': 'bias', 'sigma': 'sigma'})
        self.assertEqual(noise_template.prms_dim, 2)
        noise_template = NoiseModelTemplate(['b', {'sd': 'sigma'}], 'y')
        self.assertEqual(noise_template.prms_def,
                         {'b': 'b', 'sd': 'sigma'})
        self.assertEqual(noise_template.prms_dim, 2)
        self.assertEqual(noise_template.sensors, ['y'])
        self.assertEqual(noise_template.experiment_names, [])

    def test_add_experiment_names(self):
        # prepare the setup for the tests
        noise_template = NoiseModelTemplate(['bias', 'sigma'], 'y')
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
        # perform some simple tests for valid usage
        noise_template.add_experiment_names('Exp1', problem_experiments)
        self.assertEqual(noise_template.experiment_names, ['Exp1'])
        noise_template.add_experiment_names(['Exp2', 'Exp3'],
                                            problem_experiments)
        self.assertEqual(noise_template.experiment_names,
                         ['Exp1', 'Exp2', 'Exp3'])
        # now check some invalid input arguments
        with self.assertRaises(RuntimeError):
            # add an experiment, that does not have the noise model's sensors
            noise_template.add_experiment_names('Exp4', problem_experiments)
        with self.assertRaises(RuntimeError):
            # add experiments that refer to more than one forward model
            noise_template.add_experiment_names(['Exp5', 'Exp6'],
                                                problem_experiments)
        with self.assertRaises(RuntimeError):
            # adding the same experiment again
            noise_template.add_experiment_names('Exp1', problem_experiments)

    def test_error(self):
        # prepare the setup for the tests
        noise_template = NoiseModelTemplate(['bias', 'sigma'], ['y1', 'y2'])
        # in the NoiseModelTemplate class the error_function is just a dummy,
        # so we need to define it to be able to test the error-method
        noise_template.error_function =\
            lambda d1, d2: {name: d1[name] - d2[name]
                            for name in noise_template.sensors}
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
        noise_template.add_experiment_names(['Exp1', 'Exp2', 'Exp3'],
                                            problem_experiments)
        # now we can call the error-method
        computed_value = noise_template.error(model_response_dict,
                                              problem_experiments)
        expected_value = {'y1': np.array([1., 2., 3.]),
                          'y2': np.array([-1., -2., -3.])}
        # due to the numpy arrays we have to loop over everything to compare
        for sensor_name, array in computed_value.items():
            for i in range(len(array)):
                self.assertEqual(array[i], expected_value[sensor_name][i])

if __name__ == "__main__":
    unittest.main()
