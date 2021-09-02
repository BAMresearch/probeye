# standard library
import logging

# third party imports
import unittest
import numpy as np

# local imports
from probeye.definition.forward_model import ForwardModelTemplate
from probeye.definition.forward_model import Sensor
from probeye.definition.inference_problem import InferenceProblem
from probeye.inference.taralli_.noise_models import NormalNoise
from probeye.inference.taralli_.solver import run_taralli_solver


class TestProblem(unittest.TestCase):

    def test_taralli_solver(self):

        # define the forward model
        class LinRe(ForwardModelTemplate):
            def __call__(self, inp):
                x = inp['x']
                a = inp['a']
                b = inp['b']
                return {'y': a * x + b}

        # set up the problem
        problem = InferenceProblem("Linear regression")
        problem.add_parameter(
            'a', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        problem.add_parameter(
            'b', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        problem.add_parameter(
            'sigma', 'noise', prior=('uniform',  {'low': 0.1, 'high': 1}))
        problem.add_forward_model(
            "LinRe", LinRe(['a', 'b'], [Sensor("x")], [Sensor("y")]))
        problem.add_noise_model(NormalNoise('sigma', sensors='y'))

        # generate and add some simple test data
        n_tests, a_true, b_true, sigma_true = 30, 0.5, -0.5, 0.5
        x_test = np.linspace(0.0, 1.0, 30)
        y_true = a_true * x_test - b_true
        y_test = np.random.normal(loc=y_true, scale=sigma_true)
        problem.add_experiment(f'Tests', fwd_model_name="LinRe",
                               sensor_values={'x': x_test, 'y': y_test})

        # run the taralli solver with deactivated output
        logging.root.disabled = True
        run_taralli_solver(problem, n_walkers=20, n_steps=100,
                           show_sampling_progress=False)

if __name__ == "__main__":
    unittest.main()
