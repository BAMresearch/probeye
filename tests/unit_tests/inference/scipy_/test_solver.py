# standard library
import logging
import unittest

# third party imports
import numpy as np

# local imports
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.noise_model import NormalNoiseModel
from probeye.inference.scipy_.solver import ScipySolver


class TestProblem(unittest.TestCase):

    def test_scipy_solver(self):

        # define a simple linear model for this case
        class LinearModel(ForwardModelBase):
            def response(self, inp):
                return {'y': inp['m'] * inp['x'] + inp['b']}

        # define parameters with a circular dependency between loc_m and m
        problem = InferenceProblem("Problem with uninformative prior")
        problem.add_parameter('m', 'model')  # uninformative prior
        problem.add_parameter('b', 'model',
                              prior=('normal', {'loc': 1.0, 'scale': 1.0}))
        problem.add_parameter('sigma', 'noise',
                              prior=('uniform', {'low': 0.1, 'high': 0.8}))

        # add forward model and noise model
        isensor, osensor = Sensor("x"), Sensor("y")
        linear_model = LinearModel(['m', 'b'], [isensor], [osensor])
        problem.add_forward_model("LinearModel", linear_model)
        problem.add_noise_model(NormalNoiseModel(
            prms_def={'sigma': 'std'}, sensors=osensor))

        # add experimental data
        np.random.seed(1)
        x_test = np.linspace(0.0, 1.0, 10)
        y_true = linear_model(
            {isensor.name: x_test, 'm': 2.5, 'b': 1.7})[osensor.name]
        y_test = np.random.normal(loc=y_true, scale=0.5)
        problem.add_experiment(f'TestSeries_1', fwd_model_name="LinearModel",
                               sensor_values={isensor.name: x_test,
                                              osensor.name: y_test})

        # test the get_start_values method for given x0_dict
        scipy_solver = ScipySolver(problem)
        x0, x0_dict = scipy_solver.get_start_values(
            x0_dict={'m': 2.5, 'b': 1.5, 'sigma': 0.6})
        self.assertTrue(np.allclose(x0, np.array([2.5, 1.5, 0.6])))
        self.assertEqual(x0_dict, {'m': 2.5, 'b': 1.5, 'sigma': 0.6})

        # test the get_start_values method for automatic derivation
        scipy_solver = ScipySolver(problem)
        x0, x0_dict = scipy_solver.get_start_values(x0_prior='mean')
        self.assertTrue(np.allclose(x0, np.array([1.0, 1.0, 0.45])))
        expected_x0_dict = {'m': 1.0, 'b': 1.0, 'sigma': 0.45}
        for key, val in x0_dict.items():
            self.assertAlmostEqual(val, expected_x0_dict[key])

        # check that the solver can be run now; note that the additional solver
        # option is a default value, and only provided to check that this
        # argument-pipeline works
        scipy_solver.run_max_likelihood(solver_options={'jac': None})

if __name__ == "__main__":
    unittest.main()
