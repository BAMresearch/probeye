# standard library
import unittest

# third party imports
import numpy as np
import chaospy

# local imports
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.distribution import Normal, LogNormal
from probeye.definition.sensor import Sensor
from probeye.definition.inverse_problem import InverseProblem
from probeye.inference.bias.likelihood_models import (
    IndependentNormalModelError,
    EmbeddedLikelihoodBaseModel,
)
from probeye.inference.bias.solver import EmbeddedPCESolver


class TestProblem(unittest.TestCase):
    def test_embedded_pce_solver(self):

        # define the forward model
        class LinRe(ForwardModelBase):
            def __init__(self, name):
                super().__init__(name)

                self.pce_order = 1

            def interface(self):
                self.parameters = ["a", "b"]
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y", std_model="sigma")

            def response(self, inp: dict) -> dict:
                x = inp["x"]
                m = inp["a"]
                b = inp["b"]

                m = np.repeat(m, len(x))
                x = x.reshape(-1, 1)
                m = m.reshape(-1, 1)

                # define the distribution for the bias term
                b_dist = chaospy.Normal(0.0, b)
                expansion = chaospy.generate_expansion(self.pce_order, b_dist)

                # generate quadrature nodes and weights
                sparse_quads = chaospy.generate_quadrature(
                    self.pce_order, b_dist, rule="Gaussian"
                )
                # evaluate the model at the quadrature nodes
                sparse_evals = np.array(
                    [np.array((m + node) * x) for node in sparse_quads[0][0]]
                )
                # fit the polynomial chaos expansion
                fitted_sparse = chaospy.fit_quadrature(
                    expansion, sparse_quads[0], sparse_quads[1], sparse_evals
                )
                return {"y": fitted_sparse, "dist": b_dist}

        # set up the problem
        problem = InverseProblem("Linear regression")
        problem.add_parameter("a", prior=Normal(mean=3.5, std=0.5))
        problem.add_parameter("b", prior=LogNormal(mean=-1.0, std=0.5))
        problem.add_parameter("sigma", value=0.01)

        # generate and add some simple test data
        n_tests, a_true, b_true, sigma_true, seed = 5000, 4.0, 1.0, 0.01, 6174
        np.random.seed(seed)
        std_noise = np.linspace(0.2 * b_true, b_true, n_tests)
        std_noise += np.random.normal(loc=0.0, scale=sigma_true, size=n_tests)
        x_test = np.linspace(0.2, 1.0, n_tests)
        y_true = a_true * x_test
        y_test = y_true + np.random.normal(loc=0.0, scale=std_noise, size=n_tests)
        problem.add_experiment("Tests", sensor_data={"x": x_test, "y": y_test})

        # add the forward model
        problem.add_forward_model(LinRe("LinRe"), experiments="Tests")

        # add the likelihood model
        dummy_lmodel = EmbeddedLikelihoodBaseModel(
            experiment_name="Tests", l_model="independent_normal"
        )
        likelihood_model = IndependentNormalModelError(dummy_lmodel)
        problem.add_likelihood_model(likelihood_model)

        # run the emcee solver with different seeds (the 'vectorize=False' does not do
        # anything and is just for code coverage)
        n_walkers, n_steps = 10, 100

        solver_1a = EmbeddedPCESolver(problem, seed=123)
        inference_data_1a = solver_1a.run(
            n_walkers=n_walkers, n_steps=n_steps, vectorize=False
        )

        solver_2 = EmbeddedPCESolver(problem, seed=42)
        inference_data_2 = solver_2.run(
            n_walkers=n_walkers, n_steps=n_steps, vectorize=False
        )

        solver_1b = EmbeddedPCESolver(problem, seed=123)
        inference_data_1b = solver_1b.run(
            n_walkers=n_walkers, n_steps=n_steps, vectorize=False
        )

        # first, check that the sampled results make sense
        true_values = {"a": a_true, "b": b_true}
        for prm_name, mean_true in true_values.items():
            mean = solver_2.summary["mean"][prm_name]
            self.assertAlmostEqual(mean, mean_true, delta=0.05)

        # check that the results of '1a' and '2' are not similar (different seeds)
        same_results = True
        for prm_name in ["a", "b"]:
            v1 = inference_data_1a["posterior"][prm_name].values.flatten()
            v2 = inference_data_2["posterior"][prm_name].values.flatten()
            if np.alltrue(v1 != v2):
                same_results = False
                break
        self.assertTrue(not same_results)

        # check that the results of '1a' and '1b' are similar (same seeds)
        same_results = True
        for prm_name in ["a", "b"]:
            v1 = inference_data_1a["posterior"][prm_name].values.flatten()
            v2 = inference_data_1b["posterior"][prm_name].values.flatten()
            if np.alltrue(v1 != v2):
                same_results = False
                break
        self.assertTrue(same_results)


if __name__ == "__main__":
    unittest.main()
