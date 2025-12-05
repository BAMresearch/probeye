# standard library
import unittest
from unittest.mock import patch, MagicMock

# third party imports
import numpy as np

# local imports
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.distribution import Normal, Uniform
from probeye.definition.sensor import Sensor
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.inference.emcee.solver import EmceeSolver, RunnerFactory


class TestProblem(unittest.TestCase):
    def setUp(self):
        """Set up a common test problem for reuse."""
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
        self.problem = InverseProblem("Linear regression")
        self.problem.add_parameter("a", prior=Normal(mean=0, std=1))
        self.problem.add_parameter("b", prior=Normal(mean=0, std=1))
        self.problem.add_parameter("sigma", prior=Uniform(low=0.1, high=1))

        # generate and add some simple test data
        n_tests, a_true, b_true, sigma_true, seed = 5000, 0.3, -0.2, 0.1, 6174
        np.random.seed(seed)
        x_test = np.linspace(0.0, 1.0, n_tests)
        y_true = a_true * x_test + b_true
        y_test = np.random.normal(loc=y_true, scale=sigma_true)
        self.problem.add_experiment("Tests", sensor_data={"x": x_test, "y": y_test})

        # add the forward model
        self.problem.add_forward_model(LinRe("LinRe"), experiments="Tests")

        # add the likelihood model
        self.problem.add_likelihood_model(
            GaussianLikelihoodModel(experiment_name="Tests", model_error="additive")
        )

        self.true_values = {"a": a_true, "b": b_true, "sigma": sigma_true}

    def test_emcee_solver_classic_mode(self):
        """Test emcee solver with classic runner mode."""
        n_walkers, n_steps = 10, 100

        emcee_solver_1a = EmceeSolver(self.problem, seed=123)
        inference_data_1a = emcee_solver_1a.run(
            n_walkers=n_walkers, n_steps=n_steps, vectorize=False, cluster_mode="classic"
        )

        emcee_solver_2 = EmceeSolver(self.problem, seed=42)
        inference_data_2 = emcee_solver_2.run(
            n_walkers=n_walkers, n_steps=n_steps, vectorize=False, cluster_mode="classic"
        )

        emcee_solver_1b = EmceeSolver(self.problem, seed=123)
        inference_data_1b = emcee_solver_1b.run(
            n_walkers=n_walkers, n_steps=n_steps, vectorize=False, cluster_mode="classic"
        )

        # first, check that the sampled results make sense
        for prm_name, mean_true in self.true_values.items():
            mean = emcee_solver_2.summary["mean"][prm_name]
            self.assertAlmostEqual(mean, mean_true, delta=0.01)

        # check that the results of '1a' and '2' are not similar (different seeds)
        same_results = True
        for prm_name in ["a", "b", "sigma"]:
            v1 = inference_data_1a["posterior"][prm_name].values.flatten()
            v2 = inference_data_2["posterior"][prm_name].values.flatten()
            if np.alltrue(v1 != v2):
                same_results = False
                break
        self.assertTrue(not same_results)

        # check that the results of '1a' and '1b' are similar (same seeds)
        same_results = True
        for prm_name in ["a", "b", "sigma"]:
            v1 = inference_data_1a["posterior"][prm_name].values.flatten()
            v2 = inference_data_1b["posterior"][prm_name].values.flatten()
            if np.alltrue(v1 != v2):
                same_results = False
                break
        self.assertTrue(same_results)

    def test_emcee_solver_cluster_prune_mode(self):
        """Test emcee solver with cluster-prune runner mode."""
        n_walkers, n_steps = 20, 200

        emcee_solver = EmceeSolver(self.problem, seed=42)
        inference_data = emcee_solver.run(
            n_walkers=n_walkers, 
            n_steps=n_steps, 
            vectorize=False,
            cluster_mode="cluster-prune"
        )

        # Check that results are reasonable
        for prm_name, mean_true in self.true_values.items():
            mean = emcee_solver.summary["mean"][prm_name]
            self.assertAlmostEqual(mean, mean_true, delta=0.02)

        # Verify inference data structure
        self.assertIn("posterior", inference_data)
        for prm_name in ["a", "b", "sigma"]:
            self.assertIn(prm_name, inference_data["posterior"])

    def test_emcee_solver_parallel(self):
        """Test emcee solver with parallel execution."""
        n_walkers, n_steps = 10, 100

        emcee_solver = EmceeSolver(self.problem, seed=123)
        inference_data = emcee_solver.run(
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_initial_steps=100,
            parallel=True,
            n_processes=2,
            cluster_mode="classic"
        )

        # Check that results are reasonable
        for prm_name, mean_true in self.true_values.items():
            mean = emcee_solver.summary["mean"][prm_name]
            self.assertAlmostEqual(mean, mean_true, delta=0.02)

    def test_emcee_solver_with_true_values(self):
        """Test emcee solver summary with true values."""
        n_walkers, n_steps = 10, 100

        emcee_solver = EmceeSolver(self.problem, seed=123)
        inference_data = emcee_solver.run(
            n_walkers=n_walkers,
            n_steps=n_steps,
            true_values=self.true_values,
            cluster_mode="classic"
        )

        # Verify summary contains true values
        self.assertIn("true", emcee_solver.summary)
        for prm_name in ["a", "b", "sigma"]:
            self.assertIn(prm_name, emcee_solver.summary["true"])

    def test_restart_run(self):
        """Test restart functionality."""
        n_walkers, n_steps = 10, 50

        emcee_solver = EmceeSolver(self.problem, seed=123)
        inference_data_1 = emcee_solver.run(
            n_walkers=n_walkers,
            n_steps=n_steps,
            cluster_mode="classic"
        )

        # Get final state and restart
        state = emcee_solver.runner.sampler.get_last_sample()
        inference_data_2 = emcee_solver.restart_run(state, n_steps=30)

        # Verify both runs produced valid data
        self.assertIn("posterior", inference_data_1)
        self.assertIn("posterior", inference_data_2)

    def test_runner_factory_invalid_mode(self):
        """Test RunnerFactory with invalid mode."""
        with self.assertRaises(ValueError):
            RunnerFactory.create(
                mode="invalid_mode",
                solver=None,
                n_walkers=10,
                n_steps=100,
                n_initial_steps=50,
                parallel=False,
                n_processes=1
            )

    def test_runner_factory_none_mode(self):
        """Test RunnerFactory defaults to classic when mode is None."""
        emcee_solver = EmceeSolver(self.problem, seed=123)
        inference_data = emcee_solver.run(
            n_walkers=10,
            n_steps=50,
            cluster_mode=None  # Should default to "classic"
        )
        self.assertIn("posterior", inference_data)

    def test_runner_factory_empty_mode(self):
        """Test RunnerFactory defaults to classic when mode is empty string."""
        emcee_solver = EmceeSolver(self.problem, seed=123)
        inference_data = emcee_solver.run(
            n_walkers=10,
            n_steps=50,
            cluster_mode=""  # Should default to "classic"
        )
        self.assertIn("posterior", inference_data)

    def test_walker_clustering_no_jump(self):
        """Test walker clustering when no jump is detected."""
        n_walkers, n_steps = 10, 50

        emcee_solver = EmceeSolver(self.problem, seed=123)
        # Run with cluster-prune mode
        inference_data = emcee_solver.run(
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_initial_steps=30,
            cluster_mode="cluster-prune"
        )

        # Should still produce valid results even without pruning
        self.assertIn("posterior", inference_data)

    def test_walker_clustering_with_jump(self):
        """Test walker clustering when jump is detected."""
        # Create a scenario more likely to have divergent walkers
        n_walkers, n_steps = 30, 30

        emcee_solver = EmceeSolver(self.problem, seed=42)
        inference_data = emcee_solver.run(
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_initial_steps=20,
            cluster_mode="cluster-prune"
        )

        # Verify results are still valid
        self.assertIn("posterior", inference_data)

    def test_show_progress_false(self):
        """Test solver with show_progress=False."""
        emcee_solver = EmceeSolver(self.problem, seed=123, show_progress=False)
        inference_data = emcee_solver.run(
            n_walkers=10,
            n_steps=50,
            cluster_mode="classic"
        )
        self.assertIn("posterior", inference_data)

    def test_emcee_summary_statistics(self):
        """Test that emcee_summary computes all statistics correctly."""
        n_walkers, n_steps = 10, 100

        emcee_solver = EmceeSolver(self.problem, seed=123)
        emcee_solver.run(
            n_walkers=n_walkers,
            n_steps=n_steps,
            cluster_mode="classic"
        )

        # Verify all expected statistics are present
        expected_keys = ["mean", "median", "sd", "q05", "q95"]
        for key in expected_keys:
            self.assertIn(key, emcee_solver.summary)
            for prm_name in ["a", "b", "sigma"]:
                self.assertIn(prm_name, emcee_solver.summary[key])

    def test_additional_kwargs(self):
        """Test passing additional kwargs to EnsembleSampler."""
        n_walkers, n_steps = 10, 50

        emcee_solver = EmceeSolver(self.problem, seed=123)
        inference_data = emcee_solver.run(
            n_walkers=n_walkers,
            n_steps=n_steps,
            cluster_mode="classic",
            moves=None  # Additional kwarg for emcee
        )

        self.assertIn("posterior", inference_data)


if __name__ == "__main__":
    unittest.main()
