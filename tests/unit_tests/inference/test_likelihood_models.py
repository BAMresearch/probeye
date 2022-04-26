# standard library imports
import unittest

# third party imports
import numpy as np

# local imports
from probeye.definition.forward_model import ForwardModelBase
from probeye.inference.likelihood_models import ScipyLikelihoodBase
from probeye.inference.likelihood_models import CorrelatedModelError1V


class TestProblem(unittest.TestCase):
    def test_ScipyLikelihoodBase(self):
        # check the base class initialization and loglike-method's NotImplementedError
        scipy_likelihood_base = ScipyLikelihoodBase(
            prms_def={"sigma": "sigma"},
            experiment_name="Exp_1",
            forward_model=ForwardModelBase(name="_dummy_"),
            additive_measurement_error=False,
            correlation_variables=["x"],
            correlation_model="exp",
            name="L1",
        )
        with self.assertRaises(NotImplementedError):
            scipy_likelihood_base.loglike(np.ones(3), np.zeros(3), {})

    def test_CorrelatedModelError1V(self):
        # check the error raised when defining more than one correlation variable
        with self.assertRaises(RuntimeError):
            CorrelatedModelError1V(
                prms_def={"sigma": "sigma"},
                experiment_name="Exp_1",
                forward_model=ForwardModelBase(name="_dummy_"),
                additive_measurement_error=False,
                correlation_variables=["x", "y"],
                correlation_model="exp",
                name="L1",
            )


if __name__ == "__main__":
    unittest.main()
