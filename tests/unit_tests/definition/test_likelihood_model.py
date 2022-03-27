# standard library imports
import unittest

# third party imports
import numpy as np

# local imports
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.definition.sensor import Sensor


class TestProblem(unittest.TestCase):
    def test_init(self):
        # initialize an instance and check if everything is there
        like = GaussianLikelihoodModel(["bias", "sigma"], Sensor("y"))
        self.assertEqual(like.prms_def, {"bias": "bias", "sigma": "sigma"})
        self.assertEqual(like.prms_dim, 2)
        like = GaussianLikelihoodModel(["b", {"sd": "sigma"}], Sensor("y"))
        self.assertEqual(like.prms_def, {"b": "b", "sd": "sigma"})
        self.assertEqual(like.prms_dim, 2)
        self.assertEqual(like.sensor_names, ["y"])
        self.assertEqual(like.experiment_names, [])

    def test_add_experiment_names(self):
        # prepare the setup for the tests
        like = GaussianLikelihoodModel(["bias", "sigma"], Sensor("y"))
        problem_experiments = {
            "Exp1": {"sensor_values": {"x": 1, "y": 2}, "forward_model": "TestModel"},
            "Exp2": {"sensor_values": {"x": 3, "y": 4}, "forward_model": "TestModel"},
            "Exp3": {"sensor_values": {"x": 5, "y": 6}, "forward_model": "TestModel"},
            "Exp4": {"sensor_values": {"x": 7, "z": 8}, "forward_model": "TestModel"},
            "Exp5": {"sensor_values": {"x": 7, "y": 8}, "forward_model": "TestModel"},
            "Exp6": {"sensor_values": {"x": 7, "y": 8}, "forward_model": "OtherModel"},
        }
        # the following is usually done automatically when adding the noise model to the
        # inference problem
        like.problem_experiments = problem_experiments
        # perform some simple tests for valid usage
        like.add_experiments("Exp1")
        self.assertEqual(like.experiment_names, ["Exp1"])
        like.add_experiments(["Exp2", "Exp3"])
        self.assertEqual(like.experiment_names, ["Exp1", "Exp2", "Exp3"])
        # now check some invalid input arguments
        with self.assertRaises(RuntimeError):
            # add an experiment, that does not have the noise model's sensors
            like.add_experiments("Exp4")
        with self.assertRaises(RuntimeError):
            # add experiments that refer to more than one forward model
            like.add_experiments(["Exp5", "Exp6"])
        with self.assertRaises(RuntimeError):
            # adding the same experiment again
            like.add_experiments("Exp1")
        # when self.problem_experiments is not set yet, an error will be thrown when
        # trying to add experiments
        like.problem_experiments = None
        with self.assertRaises(ValueError):
            like.add_experiments("Exp1")

    def test_check_experiment_consistency(self):
        # check for the error when no experiments have been defined
        like = GaussianLikelihoodModel(["bias", "sigma"], Sensor("y"))
        with self.assertRaises(RuntimeError):
            like.check_experiment_consistency()

        # check for the error when the likelihood's model have different fwd models
        problem_experiments = {
            "Exp1": {"sensor_values": {"x": 1, "y": 2}, "forward_model": "TestModel1"},
            "Exp2": {"sensor_values": {"x": 3, "y": 4}, "forward_model": "TestModel2"},
        }
        like.problem_experiments = problem_experiments
        like.add_experiments("Exp1")
        like.add_experiments("Exp2")
        with self.assertRaises(RuntimeError):
            like.check_experiment_consistency()

    def test_process_correlation_definition(self):
        # check for the error when using an invalid correlation model
        with self.assertRaises(ValueError):
            _ = GaussianLikelihoodModel(
                ["bias", "sigma"], Sensor("y"), correlation_model="INVALID"
            )

        # check for the error when using both additive and multiplicative model error
        with self.assertRaises(RuntimeError):
            _ = GaussianLikelihoodModel(
                ["bias", "sigma"],
                Sensor("y"),
                additive_model_error=True,
                multiplicative_model_error=True,
            )

        # check for the error when using an invalid correlation variable
        with self.assertRaises(RuntimeError):
            _ = GaussianLikelihoodModel(
                ["bias", "sigma"], Sensor("y"), correlation_variables="u"
            )

        # check for the error when using an invalid string for the corr. variables
        with self.assertRaises(RuntimeError):
            _ = GaussianLikelihoodModel(
                ["bias", "sigma"], Sensor("y"), correlation_variables="xx"
            )


if __name__ == "__main__":
    unittest.main()
