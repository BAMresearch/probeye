# standard library imports
import unittest

# third party imports
import numpy as np

# local imports
from probeye.inference.scipy_.correlation_models import (
    SpatiotemporalExponentialCorrelationModel,
)


class TestProblem(unittest.TestCase):
    def test_check_prms(self):
        n = 10
        x_test = np.linspace(0.0, 1.0, n)
        position_array = np.tile(x_test.reshape((n, -1)), n)
        position_arrays = {"x": position_array}
        scm = SpatiotemporalExponentialCorrelationModel(position_arrays)
        self.assertTrue(not scm.check_prms({"l_corr": -1.0, "std": 1.0}))
        self.assertTrue(not scm.check_prms({"l_corr": 1.0, "std": -1.0}))


if __name__ == "__main__":
    unittest.main()
