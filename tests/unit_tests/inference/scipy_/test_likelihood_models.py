# standard library imports
import unittest

# third party imports
import numpy as np

# local imports
from probeye.definition.sensor import Sensor
from probeye.inference.scipy_.likelihood_models import AdditiveUncorrelatedModelError


class TestProblem(unittest.TestCase):
    def test_normal_noise_init(self):
        x_test = np.linspace(0.0, 1.0, 10)
        osensor = Sensor("y", x=x_test)
        with self.assertRaises(ValueError):
            AdditiveUncorrelatedModelError(
                [{"s": "std_model"}, "l_corr"],
                osensor,
                correlation_variables="x",
                correlation_model="wrong model spec",
            ).check_correlation_definition()


if __name__ == "__main__":
    unittest.main()
