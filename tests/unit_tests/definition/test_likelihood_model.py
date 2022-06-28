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

    def test_no_correlation_variables(self):
        glm = GaussianLikelihoodModel(experiment_name="Exp", model_error="additive")
        self.assertEqual(glm.correlation_variables, [])


if __name__ == "__main__":
    unittest.main()
