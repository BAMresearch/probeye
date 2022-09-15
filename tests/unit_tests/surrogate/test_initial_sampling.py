# standard library
import unittest

# third party imports
import numpy as np

# local imports
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.metamodeling.initial_sampling import LatinHypercubeSampler


class LinearModel(ForwardModelBase):
    def interface(self):
        self.parameters = ["a", "b"]
        self.input_sensors = Sensor("x")
        self.output_sensors = Sensor("y", std_model="sigma")

    def response(self, inp: dict) -> dict:
        x = inp["x"]
        m = inp["a"]
        b = inp["b"]
        return {"y": m * x + b}


class TestProblem(unittest.TestCase):
    def test_LatinHypercubeSampler_generate_samples_normal_case(self):

        # prepare a dummy problem as far as necessary for this test
        problem = InverseProblem("Test")
        problem.add_parameter("a", prior=("normal", {"mean": 2.0, "std": 1.0}))
        problem.add_parameter("b", prior=("normal", {"mean": 1.0, "std": 0.5}))

        # create the samples from the priors
        lhs = LatinHypercubeSampler(problem)
        sample_array = lhs.generate_samples(10, seed=1)

        # check if everything is as expected
        self.assertEqual(sample_array.shape, (10, 2))
        self.assertAlmostEqual(float(np.mean(sample_array[:, 0])), 2.0, delta=0.3)
        self.assertAlmostEqual(float(np.std(sample_array[:, 0])), 1.0, delta=0.3)
        self.assertAlmostEqual(float(np.mean(sample_array[:, 1])), 1.0, delta=0.3)
        self.assertAlmostEqual(float(np.std(sample_array[:, 1])), 0.5, delta=0.2)

    def test_LatinHypercubeSampler_generate_samples_mv_prior(self):

        # prepare a dummy problem as far as necessary for this test
        problem = InverseProblem("Test")
        problem.add_parameter(
            "ab",
            dim=2,
            prior=(
                "multivariate-normal",
                {
                    "mean": np.array([2.0, 1.0]),
                    "cov": np.array([[1.0, 0], [0, 0.5**2]]),
                },
            ),
        )

        # try creating the samples from the priors; this should not work since the
        # problem contains a multivariate prior
        lhs = LatinHypercubeSampler(problem)
        with self.assertRaises(RuntimeError):
            _ = lhs.generate_samples(10, seed=1)

    def test_LatinHypercubeSampler_generate_training_data(self):

        # prepare a dummy problem as far as necessary for this test
        problem = InverseProblem("Test")
        problem.add_parameter("a", prior=("normal", {"mean": 2.0, "std": 1.0}))
        problem.add_parameter("b", prior=("normal", {"mean": 1.0, "std": 0.5}))
        forward_model = LinearModel("LinearModel")
        problem.add_forward_model(forward_model)
        problem.add_experiment(
            exp_name="Exp1",
            fwd_model_name="LinearModel",
            sensor_values={
                "x": np.array([0.0, 1.0, 2.0]),
                "y": np.array([1.1, 2.9, 5.3]),
            },
        )

        # create the training data
        lhs = LatinHypercubeSampler(problem)
        prm_samples_pd, responses_over_experiments = lhs.generate_training_data(
            forward_model, 10, seed=1
        )

        # check if everything is as expected (with prm_samples_pd)
        sample_array = prm_samples_pd.values
        self.assertEqual(sample_array.shape, (10, 2))
        self.assertTrue((prm_samples_pd.columns == ["a", "b"]).all())
        self.assertAlmostEqual(float(np.mean(sample_array[:, 0])), 2.0, delta=0.3)
        self.assertAlmostEqual(float(np.std(sample_array[:, 0])), 1.0, delta=0.3)
        self.assertAlmostEqual(float(np.mean(sample_array[:, 1])), 1.0, delta=0.3)
        self.assertAlmostEqual(float(np.std(sample_array[:, 1])), 0.5, delta=0.2)

        # check if everything is as expected (with responses_over_experiments)
        self.assertEqual(responses_over_experiments["Exp1"].shape, (10, 1, 3))


if __name__ == "__main__":
    unittest.main()
