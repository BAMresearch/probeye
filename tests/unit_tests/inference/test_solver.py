# standard library imports
import unittest

# local imports
from probeye.definition.inverse_problem import InverseProblem
from probeye.inference.solver import Solver


class TestProblem(unittest.TestCase):
    def test_solver(self):
        problem = InverseProblem("Some problem")

        with self.assertRaises(NotImplementedError):
            Solver(problem)

        class Solver1(Solver):
            def _translate_parameters(self):
                pass

        with self.assertRaises(NotImplementedError):
            Solver1(problem)

        class Solver2(Solver):
            def _translate_parameters(self):
                pass

            def _translate_experiments(self):
                pass

        with self.assertRaises(NotImplementedError):
            Solver2(problem)

        class Solver3(Solver):
            def _translate_parameters(self):
                pass

            def _translate_experiments(self):
                pass

            def _translate_forward_models(self):
                pass

        with self.assertRaises(NotImplementedError):
            Solver3(problem)


if __name__ == "__main__":
    unittest.main()
