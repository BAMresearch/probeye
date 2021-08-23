# third party imports
import unittest

# local imports
from probeye.parameter import Parameters
from probeye.parameter import ParameterProperties


class TestProblem(unittest.TestCase):

    def test_parameters(self):
        # this is the standard way to use Parameters
        parameters = Parameters()
        parameters['a'] = ParameterProperties({'index': 0,
                                               'type': 'model',
                                               'role': 'const',
                                               'prior': None,
                                               'value': 1.0,
                                               'info': "...",
                                               'tex': r"$a$"})
        parameters['b'] = ParameterProperties({'index': 1,
                                               'type': 'model',
                                               'role': 'const',
                                               'prior': None,
                                               'value': 2.0,
                                               'info': "...",
                                               'tex': r"$b$"})
        parameters['c'] = ParameterProperties({'index': 2,
                                               'type': 'model',
                                               'role': 'calibration',
                                               'prior': (),
                                               'value': None,
                                               'info': "...",
                                               'tex': r"$c$"})
        # check for the parameter-role bookkeeping
        self.assertEqual(parameters.n_calibration_prms, 1)
        self.assertEqual(parameters.n_constant_prms, 2)
        self.assertEqual(parameters.n_prms, 3)
        # you cannot add the parameter's properties as a dictionary
        with self.assertRaises(ValueError):
            parameters['d'] = {'index': 3,
                               'type': 'model',
                               'role': 'calibration',
                               'prior': (),
                               'value': None,
                               'info': "...",
                               'tex': r"$d$"}

if __name__ == "__main__":
    unittest.main()
