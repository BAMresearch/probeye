# standard library
import unittest

# local imports
from probeye.definition.inference_problem import InferenceProblem


class TestProblem(unittest.TestCase):
    def test_inference_problem(self):
        # the main class was renamed to 'InverseProblem'; when using it now under the
        # old name an error is raised
        with self.assertRaises(NotImplementedError):
            InferenceProblem("TestProblem")


if __name__ == "__main__":
    unittest.main()
