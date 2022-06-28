# standard library imports
import unittest

# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports
from probeye.definition.parameter import Parameters, ParameterProperties
from probeye.inference.scipy.distributions import LogNormalDistribution
from probeye.inference.scipy.distributions import WeibullDistribution
from probeye.inference.scipy.distributions import SampleBasedDistribution

# to avoid repeating this
def value(v):
    prms_dict = {
        "index": None,
        "type": "prior",
        "domain": None,
        "prior": None,
        "value": v,
        "info": "",
        "tex": "",
    }
    return ParameterProperties(prms_dict)


class TestProblem(unittest.TestCase):
    def test_lognormal(self):
        dist = LogNormalDistribution({"mean": 2.0, "std": 1.0})
        _, ax = plt.subplots(1, 1)
        prms = Parameters()
        prms["mean_a"] = value(1.0)
        prms["std_a"] = value(2.0)
        dist.plot("a", ax, prms)
        dist.plot("a", ax, prms, x=np.linspace(0, 1, 100), color="blue", rotate=True)
        computed_result = dist("a", {"mean_a": 2.0, "std_a": 1.0}, "median", False)
        expected_result = np.exp(2.0)
        self.assertAlmostEqual(computed_result, expected_result)

    def test_weibull(self):
        dist = WeibullDistribution({"scale": 3.0, "shape": 5.0})
        _, ax = plt.subplots(1, 1)
        prms = Parameters()
        prms["scale_a"] = value(3.0)
        prms["shape_a"] = value(5.0)
        dist.plot("a", ax, prms)
        dist.plot("a", ax, prms, x=np.linspace(0, 6, 100), color="blue", rotate=True)
        computed_result = dist("a", {"scale_a": 3.0, "shape_a": 5.0}, "median", False)
        expected_result = 3.0 * np.log(2.0) ** (1.0 / 5.0)
        self.assertAlmostEqual(computed_result, expected_result)

    def test_sample_based(self):
        samples = np.random.normal(0, 1, 100)
        dist = SampleBasedDistribution({"samples": samples})
        _, ax = plt.subplots(1, 1)
        prms = Parameters()
        prms["samples_a"] = value(samples)
        dist.plot("a", ax, prms)
        dist.plot("a", ax, prms, x=np.linspace(-2, 2, 100), color="blue", rotate=True)
        computed_result = dist("a", {"samples_a": samples}, "median", False)
        expected_result = float(np.median(samples))
        self.assertAlmostEqual(computed_result, expected_result)
        computed_result = dist("a", {"samples_a": samples}, "mean", False)
        expected_result = float(np.mean(samples))
        self.assertAlmostEqual(computed_result, expected_result)


if __name__ == "__main__":
    unittest.main()
