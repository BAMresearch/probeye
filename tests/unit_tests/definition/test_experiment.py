# standard library imports
import unittest

# local imports
from probeye.definition.experiment import Experiment


class TestProblem(unittest.TestCase):
    def test_experiment(self):
        with self.assertRaises(ValueError):
            Experiment("MyExperiment", sensor_data={"T": [[1, 2], [3, 4]]})


if __name__ == "__main__":
    unittest.main()
