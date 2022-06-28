# standard library imports
import unittest

# third party imports
import numpy as np

# local imports
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.inference.scipy.likelihood_models import ScipyLikelihoodBase


class TestProblem(unittest.TestCase):
    def test_ScipyLikelihoodBase(self):
        # check the base class initialization and loglike-method's NotImplementedError
        gauss_model = GaussianLikelihoodModel(
            experiment_name="Exp", model_error="additive"
        )
        scipy_likelihood_base = ScipyLikelihoodBase(gauss_model)
        with self.assertRaises(NotImplementedError):
            scipy_likelihood_base.loglike(np.ones(3), np.zeros(3), {})


if __name__ == "__main__":
    unittest.main()
