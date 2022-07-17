# standard library imports
import unittest

# third party imports
import matplotlib.pyplot as plt

# local imports
from probeye.definition.distribution import ProbabilityDistribution
from probeye.definition.parameter import Parameters


class TestProblem(unittest.TestCase):
    def test_probability_distribution(self):
        dist = ProbabilityDistribution("normal")
        with self.assertRaises(NotImplementedError):
            dist("x", {}, "pdf")
        with self.assertRaises(NotImplementedError):
            dist.plot("x", plt.plot(), Parameters())


if __name__ == "__main__":
    unittest.main()
