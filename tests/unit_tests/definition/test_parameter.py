# third party imports
import unittest

# local imports
from probeye.definition.parameter import Parameters
from probeye.definition.parameter import ParameterProperties
from probeye.definition.prior import PriorBase


class TestProblem(unittest.TestCase):

    def test_parameters(self):
        # this is the standard way to use Parameters
        parameters = Parameters()
        parameters['a'] = ParameterProperties({'index': None,
                                               'type': 'model',
                                               'prior': None,
                                               'value': 1.0,
                                               'info': "...",
                                               'tex': r"$a$"})
        parameters['b'] = ParameterProperties({'index': None,
                                               'type': 'model',
                                               'prior': None,
                                               'value': 2.0,
                                               'info': "...",
                                               'tex': r"$b$"})
        c_prior = PriorBase('c', ['s'], 'c_dummy', 'normal')
        parameters['c'] = ParameterProperties({'index': 2,
                                               'type': 'model',
                                               'prior': c_prior,
                                               'value': None,
                                               'info': "...",
                                               'tex': r"$c$"})
        # check if the role is correctly identified
        self.assertEqual(parameters['a'].role, 'const')
        self.assertEqual(parameters['c'].role, 'latent')
        # check for the parameter-role bookkeeping
        self.assertEqual(parameters.n_latent_prms, 1)
        self.assertEqual(parameters.n_constant_prms, 2)
        self.assertEqual(parameters.n_prms, 3)
        # you cannot add the parameter's properties as a dictionary
        with self.assertRaises(ValueError):
            d_prior = PriorBase('d', ['s'], 'd_dummy', 'normal')
            parameters['d'] = {'index': 3,
                               'type': 'model',
                               'role': 'latent',
                               'prior': d_prior,
                               'value': None,
                               'info': "...",
                               'tex': r"$d$"}

        # you should not be able to set parameter attributes directly
        with self.assertRaises(AttributeError):
            parameters['a'].index = -1
        with self.assertRaises(AttributeError):
            parameters['a'].type = -1
        with self.assertRaises(AttributeError):
            parameters['a'].role = -1
        with self.assertRaises(AttributeError):
            parameters['a'].prior = -1
        with self.assertRaises(AttributeError):
            parameters['a'].value = -1

        # check if the consistency-checks work
        with self.assertRaises(TypeError):
            # index is has wrong type
            parameters['d'] = ParameterProperties({'index': True,
                                                   'type': 'model',
                                                   'prior': None,
                                                   'value': None,
                                                   'info': "...",
                                                   'tex': r"$d$"})
        with self.assertRaises(RuntimeError):
            # index has invalid value
            parameters['d'] = ParameterProperties({'index': -1,
                                                   'type': 'model',
                                                   'prior': None,
                                                   'value': 1.0,
                                                   'info': "...",
                                                   'tex': r"$d$"})
        with self.assertRaises(TypeError):
            # type has invalid type
            parameters['d'] = ParameterProperties({'index': None,
                                                   'type': True,
                                                   'prior': None,
                                                   'value': 1.0,
                                                   'info': "...",
                                                   'tex': r"$d$"})
        with self.assertRaises(RuntimeError):
            # type has invalid value
            parameters['d'] = ParameterProperties({'index': None,
                                                   'type': 'invalid value',
                                                   'prior': None,
                                                   'value': 1.0,
                                                   'info': "...",
                                                   'tex': r"$d$"})
        with self.assertRaises(TypeError):
            # prior has invalid type
            parameters['d'] = ParameterProperties({'index': None,
                                                   'type': 'model',
                                                   'prior': True,
                                                   'value': 1.0,
                                                   'info': "...",
                                                   'tex': r"$d$"})
        with self.assertRaises(TypeError):
            # value has invalid type
            parameters['d'] = ParameterProperties({'index': None,
                                                   'type': 'model',
                                                   'prior': None,
                                                   'value': True,
                                                   'info': "...",
                                                   'tex': r"$d$"})
        with self.assertRaises(TypeError):
            # info has invalid type
            parameters['d'] = ParameterProperties({'index': None,
                                                   'type': 'model',
                                                   'prior': None,
                                                   'value': 1.0,
                                                   'info': True,
                                                   'tex': r"$d$"})
        with self.assertRaises(TypeError):
            # tex has invalid type
            parameters['d'] = ParameterProperties({'index': None,
                                                   'type': 'model',
                                                   'prior': None,
                                                   'value': 1.0,
                                                   'info': "...",
                                                   'tex': True})
        with self.assertRaises(RuntimeError):
            # index and prior are inconsistently combined
            parameters['d'] = ParameterProperties({'index': 3,
                                                   'type': 'model',
                                                   'prior': None,
                                                   'value': None,
                                                   'info': "...",
                                                   'tex': r"$d$"})
        with self.assertRaises(RuntimeError):
            # index and value are inconsistently combined
            d_prior = PriorBase('d', ['s'], 'd_dummy', 'normal')
            parameters['d'] = ParameterProperties({'index': 3,
                                                   'type': 'model',
                                                   'prior': d_prior,
                                                   'value': 1.0,
                                                   'info': "...",
                                                   'tex': r"$d$"})
        with self.assertRaises(RuntimeError):
            # index and prior are inconsistently combined
            d_prior = PriorBase('d', ['s'], 'd_dummy', 'normal')
            parameters['d'] = ParameterProperties({'index': None,
                                                   'type': 'model',
                                                   'prior': d_prior,
                                                   'value': None,
                                                   'info': "...",
                                                   'tex': r"$d$"})
        with self.assertRaises(RuntimeError):
            # index and value are inconsistently combined
            parameters['d'] = ParameterProperties({'index': None,
                                                   'type': 'model',
                                                   'prior': None,
                                                   'value': None,
                                                   'info': "...",
                                                   'tex': r"$d$"})

if __name__ == "__main__":
    unittest.main()
