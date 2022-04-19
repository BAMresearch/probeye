# standard library imports
import unittest

# local imports
from probeye.definition.likelihood_model import GaussianLikelihoodModel


class TestProblem(unittest.TestCase):
    def test_init(self):
        # initialize an instance and check if everything is there
        like = GaussianLikelihoodModel(["bias", "sigma"], "Exp1")
        self.assertEqual(like.prms_def, {"bias": "bias", "sigma": "sigma"})
        self.assertEqual(like.prms_dim, 2)
        like = GaussianLikelihoodModel(["b", {"sd": "sigma"}], "Exp1")
        self.assertEqual(like.prms_def, {"b": "b", "sd": "sigma"})
        self.assertEqual(like.prms_dim, 2)

    def test_process_correlation_definition(self):
        # check for the error when using an invalid correlation model
        with self.assertRaises(ValueError):
            GaussianLikelihoodModel(
                ["bias", "sigma"], "Exp1", correlation_model="INVALID"
            ).process_correlation_definition()


if __name__ == "__main__":
    unittest.main()
