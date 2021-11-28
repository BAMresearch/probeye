# standard library imports
import unittest

# third party imports
from scipy import stats
import numpy as np

# local imports
from probeye.definition.sensor import Sensor
from probeye.inference.scipy_.noise_models import NormalNoise


class TestProblem(unittest.TestCase):
    def test_normal_noise_init(self):
        x_test = np.linspace(0.0, 1.0, 10)
        osensor = Sensor("y", x=x_test)
        with self.assertRaises(ValueError):
            NormalNoise(
                [{"s": "std"}, "l_corr"],
                osensor,
                corr_static="x",
                corr_model="wrong model spec",
                noise_type="additive",
            )


if __name__ == "__main__":
    unittest.main()
