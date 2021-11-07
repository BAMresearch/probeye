# standard library imports
import unittest

# third party imports
import numpy as np

# local imports
from probeye.inference.scipy_.correlation_models import \
    SpatialExponentialCorrelationModel


class TestProblem(unittest.TestCase):

    def test_check_prms(self):
        x_test = np.linspace(0.0, 1.0, 10)
        scm = SpatialExponentialCorrelationModel(x=x_test)
        self.assertTrue(not scm.check_prms({'l_corr': -1., 'std': 1.}))
        self.assertTrue(not scm.check_prms({'l_corr': 1., 'std': -1.}))

if __name__ == "__main__":
    unittest.main()
