# standard library
import logging
import unittest

# third party imports
import numpy as np

# local imports
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.inference.dynesty.solver import DynestySolver


class TestProblem(unittest.TestCase):
    def test_dynesty_solver(self):

        np.random.seed(6174)

        # define the forward model
        class LinRe(ForwardModelBase):
            def __call__(self, inp):
                x = inp["x"]
                a = inp["a"]
                b = inp["b"]
                return {"y": a * x + b}

        # set up the problem
        problem = InferenceProblem("Linear regression")
        problem.add_parameter("a", "model", prior=("normal", {"loc": 0, "scale": 1}))
        problem.add_parameter("b", "model", prior=("normal", {"loc": 0, "scale": 1}))
        problem.add_parameter(
            "sigma", "likelihood", prior=("uniform", {"low": 0.1, "high": 1})
        )
        problem.add_forward_model(
            "LinRe", LinRe(["a", "b"], [Sensor("x")], [Sensor("y")])
        )

        # generate and add some simple test data
        n_tests = 5000
        true = {"a": 0.3, "b": -0.2, "sigma": 0.1}
        x_test = np.linspace(0.0, 1.0, n_tests)
        y_true = true["a"] * x_test + true["b"]
        y_test = np.random.normal(loc=y_true, scale=true["sigma"])
        problem.add_experiment(
            f"Tests", fwd_model_name="LinRe", sensor_values={"x": x_test, "y": y_test}
        )

        # add the likelihood model
        problem.add_likelihood_model(
            GaussianLikelihoodModel({"sigma": "std_model"}, sensors=Sensor("y"))
        )

        # run the dynesty solver with deactivated output
        logging.root.disabled = True
        dynesty_solver = DynestySolver(problem, show_progress=True, seed=6174)
        with self.assertRaises(RuntimeError):
            dynesty_solver.run_dynesty(
                "invalid_method", nlive_init=10, nlive_batch=10, maxbatch=2
            )
        dynesty_solver.run_dynesty("dynamic", nlive_init=10, nlive_batch=10, maxbatch=2)

        sample_means = dynesty_solver.summary["mean"]
        for parameter, true_value in true.items():
            self.assertAlmostEqual(sample_means[parameter], true_value, delta=0.01)


if __name__ == "__main__":
    unittest.main()
