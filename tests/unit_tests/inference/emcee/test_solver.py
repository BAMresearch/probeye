# standard library
import logging
import unittest

# third party imports
import numpy as np

# local imports
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.inference.emcee.solver import EmceeSolver


class TestProblem(unittest.TestCase):
    def test_emcee_solver(self):

        np.random.seed(6174)

        # define the forward model
        class LinRe(ForwardModelBase):
            def interface(self):
                self.parameters = ["a", "b"]
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y", std_model="sigma")

            def __call__(self, inp):
                x = inp["x"]
                a = inp["a"]
                b = inp["b"]
                return {"y": a * x + b}

        # set up the problem
        problem = InverseProblem("Linear regression")
        problem.add_parameter("a", "model", prior=("normal", {"mean": 0, "std": 1}))
        problem.add_parameter("b", "model", prior=("normal", {"mean": 0, "std": 1}))
        problem.add_parameter(
            "sigma", "likelihood", prior=("uniform", {"low": 0.1, "high": 1})
        )
        problem.add_forward_model(LinRe("LinRe"))

        # generate and add some simple test data
        n_tests, a_true, b_true, sigma_true = 5000, 0.3, -0.2, 0.1
        x_test = np.linspace(0.0, 1.0, n_tests)
        y_true = a_true * x_test + b_true
        y_test = np.random.normal(loc=y_true, scale=sigma_true)
        problem.add_experiment(
            f"Tests", fwd_model_name="LinRe", sensor_values={"x": x_test, "y": y_test}
        )

        # add the likelihood model
        problem.add_likelihood_model(GaussianLikelihoodModel("sigma", "Tests"))

        # run the emcee solver with deactivated output
        logging.root.disabled = True
        emcee_solver = EmceeSolver(problem, show_progress=False, seed=6174)
        _ = emcee_solver.run_mcmc(n_walkers=20, n_steps=200, vectorize=False)
        # summary = run_emcee_postprocessing(problem, emcee_sampler,
        #                                    show_progress=True)
        # sample_means = summary['mean']
        # for mean, mean_true\
        #         in zip(sample_means, [a_true, b_true, sigma_true]):
        #     self.assertAlmostEqual(mean, mean_true, delta=0.01)


if __name__ == "__main__":
    unittest.main()
