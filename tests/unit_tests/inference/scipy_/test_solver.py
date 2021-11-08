# standard library
import logging
import unittest

# third party imports
import numpy as np

# local imports
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.noise_model import NormalNoiseModel
from probeye.inference.emcee_.solver import EmceeSolver


class TestProblem(unittest.TestCase):

    def test_scipy_solver(self):

        pass

if __name__ == "__main__":
    unittest.main()
