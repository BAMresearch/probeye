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
from probeye.inference.emcee_.solver import run_emcee_solver


class TestProblem(unittest.TestCase):

    def test_emcee_solver(self):

        np.random.seed(6174)

        # define the forward model
        class LinRe(ForwardModelBase):
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
        problem.add_noise_model(NormalNoiseModel({'sigma': 'std'}, sensors='y'))

        # generate and add some simple test data
        n_tests, a_true, b_true, sigma_true = 5000, 0.3, -0.2, 0.1
        x_test = np.linspace(0.0, 1.0, n_tests)
        y_true = a_true * x_test + b_true
        y_test = np.random.normal(loc=y_true, scale=sigma_true)
        problem.add_experiment(f'Tests', fwd_model_name="LinRe",
                               sensor_values={'x': x_test, 'y': y_test})

        # run the emcee solver with deactivated output
        logging.root.disabled = True
        _ = run_emcee_solver(problem, n_walkers=20, n_steps=200,
                             verbose=False, seed=6174)
        # summary = run_emcee_postprocessing(problem, emcee_sampler,
        #                                    verbose=True)
        # sample_means = summary['mean']
        # for mean, mean_true\
        #         in zip(sample_means, [a_true, b_true, sigma_true]):
        #     self.assertAlmostEqual(mean, mean_true, delta=0.01)

if __name__ == "__main__":
    unittest.main()
