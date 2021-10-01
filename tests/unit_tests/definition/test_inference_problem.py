# standard library
import io
import sys

# third party imports
import unittest
import numpy as np

# local imports
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.noise_model import NoiseModelBase


class TestProblem(unittest.TestCase):

    def test_properties(self):
        p = InferenceProblem("TestProblem")
        # add first latent parameter; n_latent_prms should be 1
        p.add_parameter('a', 'model',
                        prior=('normal', {'loc': 0.0, 'scale': 1.0}))
        self.assertEqual(p.n_latent_prms, 1)
        # add second latent parameter; n_latent_prms should be 2
        p.add_parameter('b', 'noise',
                        prior=('normal', {'loc': 0.0, 'scale': 1.0}))
        self.assertEqual(p.n_latent_prms, 2)
        # check the different properties
        self.assertEqual(p.n_prms, 6)
        self.assertEqual(
            set(p.prms), {'loc_a', 'scale_a', 'a', 'loc_b', 'scale_b', 'b'})
        self.assertEqual(p.n_latent_prms, 2)
        self.assertEqual(set(p.latent_prms), {'a', 'b'})
        self.assertEqual(p.n_constant_prms, 4)
        self.assertEqual(set(p.constant_prms),
                         {'loc_a', 'scale_a', 'loc_b', 'scale_b'})
        self.assertEqual(p.n_model_prms, 1)
        self.assertEqual(set(p.model_prms), {'a'})
        self.assertEqual(p.n_prior_prms, 4)
        self.assertEqual(set(p.prior_prms),
                         {'loc_a', 'scale_a', 'loc_b', 'scale_b'})
        self.assertEqual(p.n_noise_prms, 1)
        self.assertEqual(set(p.noise_prms), {'b'})

    def test_info(self):
        # simply check that no errors occur when info-method is called; the
        # output that is usually printed is redirected
        p = InferenceProblem("TestProblem")
        p.add_parameter('a', 'model', const=1.0)
        p.add_parameter('b', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        p.add_parameter('s', 'noise', prior=('normal', {'loc': 0, 'scale': 1}))
        sys.stdout = io.StringIO()
        # try out different options
        p.info(print_it=True, include_experiments=False, tablefmt="presto",
               check_consistency=False)
        _ = p.info(print_it=False, include_experiments=False, tablefmt="presto",
                   check_consistency=False)
        p.info(print_it=True, include_experiments=True, tablefmt="presto",
               check_consistency=False)
        p.info(print_it=True, include_experiments=True, tablefmt="plain",
               check_consistency=False)
        sys.stdout = sys.__stdout__  # reset printout to console
        with self.assertRaises(AssertionError):
            # the problem is not consistent yet (e.g. no forward model defined
            # yet), so the consistency_check will raise an error
            p.info(print_it=True, include_experiments=True, tablefmt="presto",
                   check_consistency=True)
        # now add the remaining stuff to make to problem consistent
        test_model = ForwardModelBase('b', Sensor('x'), Sensor('y'))
        p.add_forward_model('TestModel', test_model)
        p.add_noise_model(NoiseModelBase('normal', 's', sensors='y'))
        p.add_experiment('Experiment_1', sensor_values={'x': 1, 'y': 1},
                         fwd_model_name='TestModel')
        sys.stdout = io.StringIO()
        # now, the consistency_check should not raise an error
        p.info(print_it=True, include_experiments=True, tablefmt="presto",
               check_consistency=True)
        sys.stdout = sys.__stdout__  # reset printout to console

    def test_str(self):
        # set up a consistent problem and print it
        p = InferenceProblem("TestProblem")
        p.add_parameter('a', 'model', const=1.0)
        p.add_parameter('b', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        p.add_parameter('s', 'noise', prior=('normal', {'loc': 0, 'scale': 1}))
        test_model = ForwardModelBase('b', Sensor('x'), Sensor('y'))
        p.add_forward_model('TestModel', test_model)
        p.add_noise_model(NoiseModelBase('normal', 's', sensors='y'))
        p.add_experiment('Experiment_1', sensor_values={'x': 1, 'y': 1},
                         fwd_model_name='TestModel')
        sys.stdout = io.StringIO()  # redirect output to console
        print(p)
        sys.stdout = sys.__stdout__  # reset printout to console

    def test_add_parameter(self):
        p = InferenceProblem("TestProblem")
        # check valid use cases for constant parameters
        p.add_parameter('c', 'model', const=1.0,
                        info='info', tex=r'$c$')
        p.add_parameter('loc_a', 'prior', const=1.0,
                        info='info', tex=r'$loc_a$')
        p.add_parameter('sigma_1', 'noise', const=1.0,
                        info='info', tex=r'$\sigma_1$')
        # check valid use cases for latent parameters
        p.add_parameter('b', 'model', prior=('normal', {'loc': 0, 'scale': 1}),
                        info='info', tex='$b$')
        p.add_parameter('scale_a', 'prior', info='info', tex='$b$',
                        prior=('uniform', {'low': 0.1, 'high': 2.0}))
        p.add_parameter('sigma_2', 'noise', info='info', tex=r'$\sigma_2$',
                        prior=('uniform', {'low': 0.1, 'high': 2.0}))
        p.add_parameter('a', 'model', info='info', tex='$a$',
                        prior=('normal', {'loc': 'loc_a', 'scale': 'scale_a'}))
        # check invalid input arguments
        with self.assertRaises(RuntimeError):
            # adding a parameter with wrong type
            p.add_parameter('d', 'wrong_type', const=1.0)
        with self.assertRaises(RuntimeError):
            # adding a parameter with neither const nor prior given
            p.add_parameter('d', 'model')
        with self.assertRaises(RuntimeError):
            # adding a parameter with both const and prior given
            p.add_parameter('a', 'model', const=1.0,
                            prior=('uniform', {'low': 0.1, 'high': 2.0}))
        with self.assertRaises(RuntimeError):
            # adding a parameter with a name that was used before
            p.add_parameter('c', 'noise', const=2.0,
                            info='info', tex=r'$c$')
        with self.assertRaises(TypeError):
            # adding a parameter with an invalid prior
            p.add_parameter('eps', 'model', info='info', tex=r'$\epsilon$',
                            prior='Wrong prior format')
        with self.assertRaises(RuntimeError):
            # adding a parameter with an invalid prior
            p.add_parameter('eps', 'model', info='info', tex=r'$\epsilon$',
                            prior=[{'low': 0.1, 'high': 2.0}])
        with self.assertRaises(RuntimeError):
            # adding a parameter with an invalid prior
            p.add_parameter('eps', 'model', info='info', tex=r'$\epsilon$',
                            prior=('uniform', {'low': 0.1, 'high': 2.0}, '3'))
        with self.assertRaises(TypeError):
            # adding a parameter with an invalid prior
            p.add_parameter('eps', 'model', info='info', tex=r'$\epsilon$',
                            prior=(None, {'low': 0.1, 'high': 2.0}))
        with self.assertRaises(TypeError):
            # adding a parameter with an invalid prior
            p.add_parameter('eps', 'model', info='info', tex=r'$\epsilon$',
                            prior=('uniform', [{'low': 0.1}, {'high': 2.0}]))
        with self.assertRaises(TypeError):
            # adding a parameter with an invalid prior
            p.add_parameter('eps', 'model', info='info', tex=r'$\epsilon$',
                            prior=('uniform', {'low': None, 'high': 2.0}))

    def test_remove_parameter(self):
        # set up a problem with some parameters so we can remove them
        p = InferenceProblem("TestProblem")
        p.add_parameter('c', 'model', const=1.0,
                        info='info', tex=r'$c$')
        p.add_parameter('loc_a', 'prior', const=1.0,
                        info='info', tex=r'$loc_a$')
        p.add_parameter('s', 'noise', const=1.0,
                        info='info', tex=r'$\sigma$')
        p.add_parameter('scale_a', 'prior', info='info', tex='$b$',
                        prior=('uniform', {'low': 0.1, 'high': 2.0}))
        p.add_parameter('sigma', 'noise', info='info', tex=r'$\sigma$',
                        prior=('uniform', {'low': 0.1, 'high': 2.0}))
        p.add_parameter('a', 'model', info='info', tex='$a$',
                        prior=('normal', {'loc': 'loc_a', 'scale': 'scale_a'}))
        # check removing a constant parameter
        self.assertEqual(set(p.constant_prms),
                         {'c', 'loc_a', 's', 'low_scale_a', 'high_scale_a',
                          'low_sigma', 'high_sigma'})
        p.remove_parameter('c')  # <-- this is where the removal happens
        self.assertEqual(
            set(p.constant_prms), {'loc_a', 's', 'low_scale_a', 'high_scale_a',
                                   'low_sigma', 'high_sigma'})
        # check removing a latent parameter; note that removing a latent
        # parameter leads to the removal of all its prior parameters
        self.assertEqual(set(p.latent_prms), {'scale_a', 'sigma', 'a'})
        # check that indexes are as expected
        self.assertEqual(p.parameters['scale_a'].index, 0)
        self.assertEqual(p.parameters['sigma'].index, 1)
        self.assertEqual(p.parameters['a'].index, 2)
        p.remove_parameter('a')  # <-- this is where the removal happens
        # check that all latent parameters associated with 'a' are gone
        self.assertEqual(set(p.latent_prms), {'sigma'})
        # check if the re-indexing worked
        self.assertEqual(p.parameters['sigma'].index, 0)
        # check that all constant prior-parameters associated with 'a' are gone
        self.assertEqual(set(p.constant_prms), {'s', 'low_sigma', 'high_sigma'})
        # check that the removal of non-existing parameters results in an error
        with self.assertRaises(RuntimeError):
            p.remove_parameter('undefined')

    def test_check_if_parameter_exists(self):
        # check confirming existing parameter
        p = InferenceProblem("TestProblem")
        p.add_parameter('a', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        p.check_if_parameter_exists('a')
        # check RuntimeError for non-existing parameter
        with self.assertRaises(RuntimeError):
            p.check_if_parameter_exists('b')

    def test_change_parameter_role(self):
        p = InferenceProblem("TestProblem")
        # check change of role from latent to constant parameter
        p.add_parameter('a', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        self.assertEqual(set(p.latent_prms), {'a'})
        self.assertEqual(set(p.constant_prms), {'loc_a', 'scale_a'})
        self.assertEqual(set(p.prms), {'a', 'loc_a', 'scale_a'})
        p.change_parameter_role('a', const=1.0)  # <-- here the role changes
        self.assertEqual(set(p.latent_prms), set())
        self.assertEqual(set(p.constant_prms), {'a'})
        self.assertEqual(set(p.prms), {'a'})
        # check change of role from latent to constant parameter
        p.change_parameter_role('a', prior=('normal', {'loc': 0, 'scale': 1}))
        self.assertEqual(set(p.latent_prms), {'a'})
        self.assertEqual(set(p.constant_prms), {'loc_a', 'scale_a'})
        self.assertEqual(set(p.prms), {'a', 'loc_a', 'scale_a'})
        # check invalid input arguments
        with self.assertRaises(RuntimeError):
            # stated parameter does not exist
            p.change_parameter_role('undefined', const=1.0)
        with self.assertRaises(RuntimeError):
            # neither const nor prior are given
            p.change_parameter_role('a')
        with self.assertRaises(RuntimeError):
            # both const and prior are given
            p.change_parameter_role(
                'a', const=1.0, prior=('normal', {'loc': 0, 'scale': 1}))
        with self.assertRaises(RuntimeError):
            # change to role the parameter already has
            p.change_parameter_role(
                'a', prior=('normal', {'loc': 0, 'scale': 1}))

    def test_change_parameter_info(self):
        p = InferenceProblem("TestProblem")
        # simple check that the change works and has the expected effect
        p.add_parameter('a', 'model', prior=('normal', {'loc': 0, 'scale': 1}),
                        info='Info', tex='$a$')
        new_info, new_tex = 'New info', r'$\hat{a}$'
        p.change_parameter_info('a', new_info, new_tex=new_tex)
        self.assertEqual(p.parameters['a'].info, new_info)
        self.assertEqual(p.parameters['a'].tex, new_tex)
        # check call for non-existing parameter
        with self.assertRaises(RuntimeError):
            p.change_parameter_info('b', new_info, new_tex=new_tex)

    def test_change_constant(self):
        p = InferenceProblem("TestProblem")
        # simple check that the change works and has the expected effect
        p.add_parameter('a', 'model', const=0.0)
        p.add_parameter('b', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        new_const = 1.0
        p.change_constant('a', new_const)
        self.assertEqual(p.parameters['a'].value, new_const)
        # check for invalid input arguments
        with self.assertRaises(RuntimeError):
            # given parameter does not exist
            p.change_constant('undefined', 2.0)
        with self.assertRaises(RuntimeError):
            # given parameter is not a constant parameter
            p.change_constant('b', 2.0)

    def test_add_prior(self):
        # check valid use case
        p = InferenceProblem("TestProblem")
        p.add_parameter('loc_a', 'prior', const=0.0)
        p.add_parameter('scale_a', 'prior', const=1.0)
        _ = p._add_prior('a_normal', 'normal', ['loc_a', 'scale_a'], 'a')
        # check invalid use cases
        with self.assertRaises(RuntimeError):
            # add a prior with an undefined parameter
            _ = p._add_prior('a_normal', 'normal', ['undef', 'scale_a'], 'a')
        with self.assertRaises(RuntimeError):
            # add a prior with a similar name
            _ = p._add_prior('a_normal', 'normal', ['loc_a', 'scale_a'], 'a')

    def test_check_problem_consistency(self):
        p = InferenceProblem("TestProblem")
        # check if the method identifies inconsistent problems
        with self.assertRaises(AssertionError):
            # nothing is defined at this point
            p.check_problem_consistency()
        # define some parameters that do not require priors
        p.add_parameter('loc_a', 'prior', const=0.0)
        p.add_parameter('scale_a', 'prior', const=1.0)
        with self.assertRaises(AssertionError):
            # no priors defined yet
            p.check_problem_consistency()
        # add a prior by defining a latent parameter
        p.add_parameter('a', 'model', info='Some model parameter', tex='$a$',
                        prior=('normal', {'loc': 'loc_a', 'scale': 'scale_a'}))
        with self.assertRaises(AssertionError):
            # no forward models defined yet
            p.check_problem_consistency()
        # add a forward model
        test_model = ForwardModelBase('a', Sensor('x'), Sensor('y'))
        p.add_forward_model('TestModel', test_model)
        with self.assertRaises(AssertionError):
            # no noise models defined yet
            p.check_problem_consistency()
        # add a noise model
        p.add_parameter('s', 'noise', const=1.0)
        noise_model = NoiseModelBase('normal', 's', sensors=['y'])
        p.add_noise_model(noise_model)
        with self.assertRaises(AssertionError):
            # no experiment_names defined yet
            p.check_problem_consistency()
        # add at an experiment
        p.add_experiment('Experiment_1', sensor_values={'x': 1, 'y': 1},
                         fwd_model_name='TestModel')
        # now the problem should be consistent
        p.check_problem_consistency()

    def test_add_experiment(self):
        # check correct use
        p = InferenceProblem("TestProblem")
        p.add_parameter('a', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        test_model = ForwardModelBase('a', Sensor('x'), Sensor('y'))
        p.add_forward_model('TestModel', test_model)
        p.add_experiment('Experiment_1', sensor_values={'x': 1, 'y': 1},
                         fwd_model_name='TestModel')
        # the experiment may contain more sensors than needed by forward model
        p.add_experiment('Experiment_2', sensor_values={'x': 1, 'y': 1, 'z': 1},
                         fwd_model_name='TestModel')
        # check that scalar sensor_values are not transformed to numpy arrays
        x_in_p = p.experiments['Experiment_2']['sensor_values']['x']
        self.assertEqual(type(x_in_p), type(1))
        # check invalid input arguments
        with self.assertRaises(TypeError):
            # wrong sensor_values type
            # noinspection PyTypeChecker
            p.add_experiment('Experiment_3', sensor_values=[('x', 1), ('y', 2)],
                             fwd_model_name='TestModel')
        with self.assertRaises(TypeError):
            # wrong fwd_model_name type
            # noinspection PyTypeChecker
            p.add_experiment('Experiment_3', sensor_values={'x': 1, 'y': 1},
                             fwd_model_name=1.2)
        with self.assertRaises(RuntimeError):
            # referencing non-existing forward model
            p.add_experiment('Experiment_3', sensor_values={'x': 1, 'y': 1},
                             fwd_model_name='ThisModelDoesNotExist')
        with self.assertRaises(RuntimeError):
            # forward model's input sensor not provided by experiment
            p.add_experiment('Experiment_3', sensor_values={'y': 1},
                             fwd_model_name='TestModel')
        with self.assertRaises(RuntimeError):
            # forward model's output sensor not provided by experiment
            p.add_experiment('Experiment_3', sensor_values={'x': 1},
                             fwd_model_name='TestModel')
        with self.assertRaises(RuntimeError):
            # sensor values with different lengths
            p.add_experiment('Experiment_3', fwd_model_name='TestModel',
                             sensor_values={'x': [1, 2], 'y': 1})
        # check that sensor_value lists are transformed to numpy arrays
        p.add_experiment('Experiment_3', fwd_model_name='TestModel',
                         sensor_values={'x': [1, 2], 'y': [2, 3]})
        x_in_p = p.experiments['Experiment_3']['sensor_values']['x']
        y_in_p = p.experiments['Experiment_3']['sensor_values']['y']
        self.assertEqual(type(x_in_p), np.ndarray)
        self.assertEqual(type(y_in_p), np.ndarray)
        # check adding the same experiment again; note that this results in a
        # warning that is printed; this printout is redirected below
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        p.add_experiment('Experiment_1', sensor_values={'x': 1, 'y': 1},
                         fwd_model_name='TestModel')
        sys.stdout = sys.__stdout__  # reset printout to console
        warning = capturedOutput.getvalue()
        self.assertEqual(warning, "WARNING - Experiment 'Experiment_1' is "
                                 "already defined and will be overwritten!\n")

    def test_get_parameters(self):
        # check a simple use case
        p = InferenceProblem("TestProblem")
        p.add_parameter('c', 'model', const=1.0)
        p.add_parameter('a', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        p.add_parameter('b', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        a_value, b_value = 3.1, 14.7
        prms_def = {'b': 'b', 'a': 'a', 'loc_b': 'loc_b', 'c': 'c'}
        computed_result = p.get_parameters([a_value, b_value], prms_def)
        expected_result = {'b': b_value, 'a': a_value, 'loc_b': 0, 'c': 1.0}
        self.assertEqual(computed_result, expected_result)

    def test_get_experiment_names(self):
        # prepare for checks
        p = InferenceProblem("TestProblem")
        p.add_parameter('a', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        # define two forward models
        test_model_1 = ForwardModelBase('a', Sensor('x'),
                                        [Sensor('y1'), Sensor('y2')])
        test_model_2 = ForwardModelBase('a', Sensor('x'),
                                        [Sensor('z1'), Sensor('z2')])
        p.add_forward_model('TestModel_1', test_model_1)
        p.add_forward_model('TestModel_2', test_model_2)
        # define experiment_names to each forward model
        p.add_experiment('Experiment_Y1',
                         sensor_values={'x': 1, 'y1': 1, 'y2': 1},
                         fwd_model_name='TestModel_1')
        p.add_experiment('Experiment_Y2',
                         sensor_values={'x': 2, 'y1': 2, 'y2': 2},
                         fwd_model_name='TestModel_1')
        p.add_experiment('Experiment_Z1',
                         sensor_values={'x': -1, 'z1': -1, 'z2': -1},
                         fwd_model_name='TestModel_2')
        p.add_experiment('Experiment_Z2',
                         sensor_values={'x': -2, 'z1': -2, 'z2': -2},
                         fwd_model_name='TestModel_2')
        # get experiment_names for specific forward model
        computed_result = p.get_experiment_names(
            forward_model_names='TestModel_1')
        expected_result = ['Experiment_Y1', 'Experiment_Y2']
        for i in range(len(computed_result)):
            self.assertEqual(computed_result[i], expected_result[i])
        # get experiment_names for specific forward model
        computed_result = p.get_experiment_names(
            forward_model_names='TestModel_2')
        expected_result = ['Experiment_Z1', 'Experiment_Z2']
        for i in range(len(computed_result)):
            self.assertEqual(computed_result[i], expected_result[i])
        # get experiment_names for both forward models
        computed_result = p.get_experiment_names(
            forward_model_names=['TestModel_1', 'TestModel_2'])
        expected_result = ['Experiment_Y1', 'Experiment_Y2',
                           'Experiment_Z1', 'Experiment_Z2']
        for i in range(len(computed_result)):
            self.assertEqual(computed_result[i], expected_result[i])
        # get experiment_names via output sensors
        computed_result = p.get_experiment_names(sensor_names=['y1', 'y2'])
        expected_result = ['Experiment_Y1', 'Experiment_Y2']
        for i in range(len(computed_result)):
            self.assertEqual(computed_result[i], expected_result[i])
        # get experiment_names via output sensors
        computed_result = p.get_experiment_names(sensor_names=['z1', 'z2'])
        expected_result = ['Experiment_Z1', 'Experiment_Z2']
        for i in range(len(computed_result)):
            self.assertEqual(computed_result[i], expected_result[i])
        # get experiment_names via output sensors
        computed_result = p.get_experiment_names(
            sensor_names=['z1', 'y1', 'z2'])
        expected_result = []
        self.assertEqual(computed_result, expected_result)
        # get experiment_names via both forward model name and sensors
        computed_result = p.get_experiment_names(
            sensor_names=['y1', 'y2'], forward_model_names='TestModel_1')
        expected_result = ['Experiment_Y1', 'Experiment_Y2']
        for i in range(len(computed_result)):
            self.assertEqual(computed_result[i], expected_result[i])
        # get experiment_names via both forward model name and sensors
        computed_result = p.get_experiment_names(
            sensor_names=['y1', 'y2'], forward_model_names='TestModel_2')
        expected_result = []
        self.assertEqual(computed_result, expected_result)
        # check invalid input arguments
        with self.assertRaises(RuntimeError):
            # neither forward_model_names nor sensor_names given
            p.get_experiment_names()

    def test_get_theta_names(self):
        # check some simple use cases
        p = InferenceProblem("TestProblem")
        p.add_parameter(
            'a', 'model', prior=('normal', {'loc': 0, 'scale': 1}), tex="$a$")
        p.add_parameter(
            'b', 'model', prior=('normal', {'loc': 0, 'scale': 1}), tex="$b$")
        computed_result = p.get_theta_names(tex=False)
        expected_result = ['a', 'b']
        self.assertEqual(computed_result, expected_result)
        computed_result = p.get_theta_names(tex=True)
        expected_result = ['$a$', '$b$']
        self.assertEqual(computed_result, expected_result)

    def test_theta_explanation(self):
        # simply check that no errors occur when theta_explanation is called;
        # the output that is usually printed is redirected
        p = InferenceProblem("TestProblem")
        p.add_parameter('a', 'model', const=1.0)
        p.add_parameter('b', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        p.add_parameter('s', 'noise', prior=('normal', {'loc': 0, 'scale': 1}))
        sys.stdout = io.StringIO()
        # try out different options
        p.theta_explanation(print_it=True, check_consistency=False)
        _ = p.theta_explanation(print_it=False, check_consistency=False)
        sys.stdout = sys.__stdout__  # reset printout to console
        # check the model_consistency flag
        with self.assertRaises(AssertionError):
            # the model is not consistent
            p.theta_explanation(print_it=True, check_consistency=True)

    def test_add_forward_model(self):
        # check correct use
        p = InferenceProblem("TestProblem")
        p.add_parameter('a', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        test_model = ForwardModelBase('a', Sensor('x'), Sensor('y'))
        p.add_forward_model('TestModel', test_model)
        # check for invalid input arguments
        with self.assertRaises(RuntimeError):
            # a given parameter of the forward model has not been added yet
            test_model_2 = ForwardModelBase('b', Sensor('x'), Sensor('y'))
            p.add_forward_model('TestModel_2', test_model_2)
        with self.assertRaises(RuntimeError):
            # add a forward model with the same name
            p.add_forward_model('TestModel', test_model)
        with self.assertRaises(RuntimeError):
            # add a forward model with already used output sensor name
            test_model_2 = ForwardModelBase('a', Sensor('x'), Sensor('y'))
            p.add_forward_model('TestModel_2', test_model_2)

    def test_evaluate_model_response(self):
        # prepare for checks
        p = InferenceProblem("TestProblem")
        p.add_parameter('a0', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        p.add_parameter('a1', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        p.add_parameter('a2', 'model', prior=('normal', {'loc': 0, 'scale': 1}))

        class FwdModel(ForwardModelBase):
            def __call__(self, inp):
                x = inp['x']
                a0 = inp['a0']
                a1 = inp['a1']
                a2 = inp['a2']
                return {'y': a0 + a1 * x + a2 * x ** 2}

        # add forward model
        fwd_model = FwdModel(['a0', 'a1', 'a2'], Sensor('x'), Sensor('y'))
        p.add_forward_model('FwdModel', fwd_model)
        # add experiment_names
        p.add_experiment('Exp1', sensor_values={'x': 1, 'y': 2},
                         fwd_model_name='FwdModel')
        p.add_experiment('Exp2', sensor_values={'x': 2, 'y': 3},
                         fwd_model_name='FwdModel')
        p.add_experiment('Exp3', sensor_values={'x': [1, 2], 'y': [1, 2]},
                         fwd_model_name='FwdModel')
        # perform a check for all experiments
        a0_value, a1_value, a2_value = 1, 2, 3
        theta = np.array([a0_value, a1_value, a2_value])
        computed_result = p.evaluate_model_response(theta)
        expected_result = {'Exp1': {'y': 6},
                           'Exp2': {'y': 17},
                           'Exp3': {'y': np.array([6, 17])}}
        # check for each item, because the last contains an array, which results
        # in an error is you do assertEqual on the whole thing
        self.assertEqual(computed_result['Exp1'], expected_result['Exp1'])
        self.assertEqual(computed_result['Exp2'], expected_result['Exp2'])
        self.assertEqual(computed_result['Exp3']['y'][0],
                         expected_result['Exp3']['y'][0])
        self.assertEqual(computed_result['Exp3']['y'][1],
                         expected_result['Exp3']['y'][1])
        # perform a check for a subset of experiments
        computed_result = p.evaluate_model_response(
            theta, experiment_names='Exp3')
        expected_result = {'Exp3': {'y': np.array([6, 17])}}
        self.assertEqual(computed_result['Exp3']['y'][0],
                         expected_result['Exp3']['y'][0])
        self.assertEqual(computed_result['Exp3']['y'][1],
                         expected_result['Exp3']['y'][1])

    def test_add_noise_model(self):
        # check correct use
        p = InferenceProblem("TestProblem")
        p.add_parameter('a', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        p.add_parameter('s1', 'noise', prior=('normal', {'loc': 0, 'scale': 1}))
        p.add_parameter('s2', 'noise', prior=('normal', {'loc': 0, 'scale': 1}))
        p.add_parameter('s3', 'noise', prior=('normal', {'loc': 0, 'scale': 1}))
        test_model = ForwardModelBase('a', Sensor('x'),
                                      [Sensor('y1'), Sensor('y2')])
        p.add_forward_model('TestModel', test_model)
        noise_model1 = NoiseModelBase('normal', 's1', sensors=['y1', 'y2'])
        noise_model2 = NoiseModelBase('normal', ['s2'], sensors=['y2'])
        noise_model3 = NoiseModelBase('normal', ['s1', 's2', 's3'],
                                      sensors=['y1'])
        p.add_noise_model(noise_model1)
        p.add_noise_model(noise_model2)
        p.add_noise_model(noise_model3)
        # check invalid input arguments
        with self.assertRaises(RuntimeError):
            # the given noise model parameter has not been defined
            p.add_noise_model(NoiseModelBase('normal', 'not_existing_parameter',
                                             sensors=['y1', 'y2']))

    def test_assign_experiments_to_noise_models(self):
        # some preparations before performing the actual tests
        p = InferenceProblem("TestProblem")
        # add some parameters
        p.add_parameter('a', 'model', prior=('normal', {'loc': 0, 'scale': 1}))
        p.add_parameter('s1', 'noise', prior=('normal', {'loc': 0, 'scale': 1}))
        p.add_parameter('s2', 'noise', prior=('normal', {'loc': 0, 'scale': 1}))
        p.add_parameter('s3', 'noise', prior=('normal', {'loc': 0, 'scale': 1}))
        # add some dummy forward models (only the output sensors are important)
        test_model_y1 = ForwardModelBase('a', Sensor('x'), Sensor('y1'))
        test_model_y2 = ForwardModelBase('a', Sensor('x'), Sensor('y2'))
        test_model_z1z2 = ForwardModelBase('a', Sensor('x'),
                                           [Sensor('z1'), Sensor('z2')])
        p.add_forward_model('TestModel_y1', test_model_y1)
        p.add_forward_model('TestModel_y2', test_model_y2)
        p.add_forward_model('TestModel_z1z2', test_model_z1z2)
        # add some noise models
        noise_model_y1 = NoiseModelBase('normal', ['s1', 's2', 's3'],
                                        sensors=['y1'])
        noise_model_y2 = NoiseModelBase('normal', ['s2'], sensors=['y2'])
        noise_model_y1y2 = NoiseModelBase('normal', 's1', sensors=['z1', 'z2'])
        p.add_noise_model(noise_model_y1)
        p.add_noise_model(noise_model_y2)
        p.add_noise_model(noise_model_y1y2)
        # add some experiments
        p.add_experiment('Exp_y1', sensor_values={'x': 1, 'y1': 2},
                         fwd_model_name='TestModel_y1')
        p.add_experiment('Exp_y2_1', sensor_values={'x': 3, 'y2': 4},
                         fwd_model_name='TestModel_y2')
        p.add_experiment('Exp_y2_2', sensor_values={'x': 5, 'y2': 6},
                         fwd_model_name='TestModel_y2')
        p.add_experiment('Exp_z1z2', sensor_values={'x': 7, 'z1': 8, 'z2': 9},
                         fwd_model_name='TestModel_z1z2')
        # this is the call that should be tested here
        p.assign_experiments_to_noise_models()
        # now check if all experiments have been assigned correctly
        self.assertEqual(
            p.noise_models[0].experiment_names, ['Exp_y1'])
        self.assertEqual(
            p.noise_models[1].experiment_names,  ['Exp_y2_1', 'Exp_y2_2'])
        self.assertEqual(
            p.noise_models[2].experiment_names, ['Exp_z1z2'])

if __name__ == "__main__":
    unittest.main()
