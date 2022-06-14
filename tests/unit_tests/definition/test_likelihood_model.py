# standard library imports
import unittest

# local imports
from probeye.definition.likelihood_model import GaussianLikelihoodModel


class TestProblem(unittest.TestCase):
    def test_wrong_model_error_flag(self):
        # check for the error when using an invalid model error string
        with self.assertRaises(ValueError):
            GaussianLikelihoodModel(
                experiment_name="Exp1",
                model_error="INVALID",
            )


if __name__ == "__main__":
    unittest.main()
