# standard library
import unittest

# third party imports
import numpy as np

# local imports
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.noise_model import NormalNoiseModel
from probeye.inference.torch_.solver import PyroSolver


class TestProblem(unittest.TestCase):

    def test_check_for_circular_dependency(self):

        # define a simple linear model for this case
        class LinearModel(ForwardModelBase):
            def response(self, inp):
                return {'y': inp['m'] * inp['x'] + inp['b']}

        # define parameters with a circular dependency between loc_m and m
        problem = InferenceProblem("Problem with circular dependency")
        problem.add_parameter('loc_m', 'prior',
                              prior=('uniform', {'low': 2.0, 'high': 3.0}))
        problem.add_parameter('m', 'model',
                              prior=('normal', {'loc': 'loc_m', 'scale': 1.0}))
        problem.add_parameter('b', 'model',
                              prior=('normal', {'loc': 1.0, 'scale': 1.0}))
        problem.add_parameter('sigma', 'noise',
                              prior=('uniform', {'low': 0.1, 'high': 0.8}))
        problem.remove_parameter('loc_m')
        problem.add_parameter('loc_m', 'prior',
                              prior=('uniform', {'low': 'm', 'high': 3.0}))

        # add forward model and noise model
        isensor, osensor = Sensor("x"), Sensor("y")
        linear_model = LinearModel(['m', 'b'], [isensor], [osensor])
        problem.add_forward_model("LinearModel", linear_model)
        problem.add_noise_model(NormalNoiseModel(
            prms_def={'sigma': 'std'}, sensors=osensor))

        # add experimental data
        np.random.seed(1)
        x_test = np.linspace(0.0, 1.0, 10)
        y_true = linear_model(
            {isensor.name: x_test, 'm': 2.5, 'b': 1.7})[osensor.name]
        y_test = np.random.normal(loc=y_true, scale=0.5)
        problem.add_experiment(f'TestSeries_1', fwd_model_name="LinearModel",
                               sensor_values={isensor.name: x_test,
                                              osensor.name: y_test})

        # the pre-check in PyroSolver should now detect the circular dependency
        with self.assertRaises(RuntimeError):
            PyroSolver(problem)

    def test_dependency_rearrangement(self):
        # define a simple linear model for this case
        class LinearModel(ForwardModelBase):
            def response(self, inp):
                return {'y': inp['m'] * inp['x'] + inp['b']}

        # define parameters where 'm' depends on 'loc_m', but 'loc_m' is added
        # after 'm', so that it has to be corrected for pyro to work
        problem = InferenceProblem("Problem with circular dependency")
        problem.add_parameter('loc_m', 'prior',
                              prior=('uniform', {'low': 2.0, 'high': 3.0}))
        problem.add_parameter('m', 'model',
                              prior=('normal', {'loc': 'loc_m', 'scale': 1.0}))
        problem.add_parameter('b', 'model',
                              prior=('normal', {'loc': 1.0, 'scale': 1.0}))
        problem.add_parameter('sigma', 'noise',
                              prior=('uniform', {'low': 0.1, 'high': 0.8}))
        problem.remove_parameter('loc_m')
        problem.add_parameter('loc_m', 'prior',
                              prior=('uniform', {'low': 1.0, 'high': 3.0}))

        # add forward model and noise model
        isensor, osensor = Sensor("x"), Sensor("y")
        linear_model = LinearModel(['m', 'b'], [isensor], [osensor])
        problem.add_forward_model("LinearModel", linear_model)
        problem.add_noise_model(NormalNoiseModel(
            prms_def={'sigma': 'std'}, sensors=osensor))

        # add experimental data
        np.random.seed(1)
        x_test = np.linspace(0.0, 1.0, 10)
        y_true = linear_model(
            {isensor.name: x_test, 'm': 2.5, 'b': 1.7})[osensor.name]
        y_test = np.random.normal(loc=y_true, scale=0.5)
        problem.add_experiment(f'TestSeries_1', fwd_model_name="LinearModel",
                               sensor_values={isensor.name: x_test,
                                              osensor.name: y_test})

        # here it is finally checked, that the rearrangement works
        pyro_solver = PyroSolver(problem, show_progress=False)
        computed_result = pyro_solver.dependency_dict
        expected_result = {'loc_m': [], 'b': [], 'sigma': [], 'm': ['loc_m']}
        self.assertEqual(computed_result, expected_result)
        # additionally check that the actual sampling works; note that the
        # target_accept_prob argument is stated here to check if it works to
        # pass additional keyword arguments to the run_mcmc method
        pyro_solver.run_mcmc(n_walkers=1, n_steps=100, n_initial_steps=10,
                             target_accept_prob=0.8)

        # check the warning when the problem cannot be deep-copied
        problem.no_deepcopy_possible = (i for i in (1, 2, 3))
        PyroSolver(problem)  # this should result in a warning

if __name__ == "__main__":
    unittest.main()
