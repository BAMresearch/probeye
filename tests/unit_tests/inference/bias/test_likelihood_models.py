# standard library
import math
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
    EmbeddedLikelihoodBaseModel,
    IndependentNormalModelError,
    GlobalMomentMatchingModelError,
    RelativeGlobalMomentMatchingModelError,
    MomentMatchingModelError,
)
from probeye.inference.bias.solver import EmbeddedPCESolver


class TestProblem(unittest.TestCase):
    def test_embedded_likelihood_models(self):

        # define the forward model
        class LinRe(ForwardModelBase):
            def interface(self):
                self.parameters = ["a", "b"]
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y", std_model="sigma")

            def response(self, inp: dict) -> dict:
                pce_order = 1
                x = inp["x"]
                m = inp["a"]
                b = inp["b"]

                m = np.repeat(m, len(x))
                x = x.reshape(-1, 1)
                m = m.reshape(-1, 1)

                # define the distribution for the bias term
                b_dist = chaospy.Normal(0.0, b)
                # generate quadrature nodes and weights
                sparse_quads = chaospy.generate_quadrature(
                    pce_order, b_dist, rule="Gaussian"
                )
                # evaluate the model at the quadrature nodes
                sparse_evals = np.array(
                    [np.array((m + node) * x) for node in sparse_quads[0][0]]
                )
                # generate the polynomial chaos expansion
                expansion = chaospy.generate_expansion(pce_order, b_dist)
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
        n_walkers, n_steps, n_initial_steps = 10, 100, 20

        in_solver = EmbeddedPCESolver(problem, seed=123)
        inference_data_in = in_solver.run(
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            vectorize=False,
        )

        problem.likelihood_models.pop("Tests")
        dummy_lmodel.gamma = 1.0
        dummy_lmodel.l_model = "global_moment_matching"
        likelihood_model = GlobalMomentMatchingModelError(dummy_lmodel)
        problem.add_likelihood_model(likelihood_model)
        gmm_solver = EmbeddedPCESolver(problem, seed=42)
        inference_data_gmm = gmm_solver.run(
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            vectorize=False,
        )

        problem.likelihood_models.pop("Tests")
        dummy_lmodel.l_model = "relative_global_moment_matching"
        likelihood_model = RelativeGlobalMomentMatchingModelError(dummy_lmodel)
        problem.add_likelihood_model(likelihood_model)
        rgmm_solver = EmbeddedPCESolver(problem, seed=42)
        inference_data_rgmm = rgmm_solver.run(
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            vectorize=False,
        )
        problem.likelihood_models.pop("Tests")
        dummy_lmodel.tolerance = 1e-6
        dummy_lmodel.gamma = np.sqrt(math.pi / 2)
        dummy_lmodel.l_model = "moment_matching"
        likelihood_model = MomentMatchingModelError(dummy_lmodel)
        problem.add_likelihood_model(likelihood_model)
        mm_solver = EmbeddedPCESolver(problem, seed=42)
        inference_data_mm = mm_solver.run(
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            vectorize=False,
        )

        # Check that the sampled results make sense
        true_values = {"a": a_true, "b": b_true}
        for prm_name, mean_true in true_values.items():
            mean_in = in_solver.summary["mean"][prm_name]
            self.assertAlmostEqual(mean_in, mean_true, delta=0.05)
            mean_rgmm = rgmm_solver.summary["mean"][prm_name]
            self.assertAlmostEqual(mean_rgmm, mean_true, delta=0.05)
            mean_gmm = gmm_solver.summary["mean"][prm_name]
            self.assertAlmostEqual(mean_gmm, mean_true, delta=0.05)
            mean_mm = mm_solver.summary["mean"][prm_name]
            self.assertAlmostEqual(mean_mm, mean_true, delta=0.05)


if __name__ == "__main__":
    unittest.main()
