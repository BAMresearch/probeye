# standard library
import unittest

# third party imports
import torch as th
import numpy as np

# local imports
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.inference.torch.solver import PyroSolver


class TestProblem(unittest.TestCase):
    def test_check_for_circular_dependency(self):

        # define a simple linear model for this case
        class LinearModel(ForwardModelBase):
            def response(self, inp):
                return {"y": inp["m"] * inp["x"] + inp["b"]}

        # define parameters with a circular dependency between loc_m and m
        problem = InferenceProblem("Problem with circular dependency")
        problem.add_parameter(
            "loc_m", "prior", prior=("uniform", {"low": 2.0, "high": 3.0})
        )
        problem.add_parameter(
            "m", "model", prior=("normal", {"loc": "loc_m", "scale": 1.0})
        )
        problem.add_parameter(
            "b", "model", prior=("normal", {"loc": 1.0, "scale": 1.0})
        )
        problem.add_parameter(
            "sigma", "likelihood", prior=("uniform", {"low": 0.1, "high": 0.8})
        )
        problem.remove_parameter("loc_m")
        problem.add_parameter(
            "loc_m", "prior", prior=("uniform", {"low": "m", "high": 3.0})
        )

        # add forward model
        isensor, osensor = Sensor("x"), Sensor("y")
        linear_model = LinearModel(["m", "b"], [isensor], [osensor])
        problem.add_forward_model("LinearModel", linear_model)

        # add experimental data
        np.random.seed(1)
        x_test = np.linspace(0.0, 1.0, 10)
        y_true = linear_model({isensor.name: x_test, "m": 2.5, "b": 1.7})[osensor.name]
        y_test = np.random.normal(loc=y_true, scale=0.5)
        problem.add_experiment(
            f"TestSeries_1",
            fwd_model_name="LinearModel",
            sensor_values={isensor.name: x_test, osensor.name: y_test},
        )

        # add the likelihood model
        problem.add_likelihood_model(
            GaussianLikelihoodModel(prms_def={"sigma": "std_model"}, sensors=osensor)
        )

        # the pre-check in PyroSolver should now detect the circular dependency
        with self.assertRaises(RuntimeError):
            PyroSolver(problem)

    def test_dependency_rearrangement(self):
        # define a simple linear model for this case
        class LinearModel(ForwardModelBase):
            def response(self, inp):
                return {"y": inp["m"] * inp["x"] + inp["b"]}

        # define parameters where 'm' depends on 'loc_m', but 'loc_m' is added after
        # 'm', so that it has to be corrected for pyro to work
        problem = InferenceProblem("Problem with circular dependency")
        problem.add_parameter(
            "loc_m", "prior", prior=("uniform", {"low": 2.0, "high": 3.0})
        )
        problem.add_parameter(
            "m", "model", prior=("normal", {"loc": "loc_m", "scale": 1.0})
        )
        problem.add_parameter(
            "b", "model", prior=("normal", {"loc": 1.0, "scale": 1.0})
        )
        problem.add_parameter(
            "sigma", "likelihood", prior=("uniform", {"low": 0.1, "high": 0.8})
        )
        problem.remove_parameter("loc_m")
        problem.add_parameter(
            "loc_m", "prior", prior=("uniform", {"low": 1.0, "high": 3.0})
        )

        # add forward model
        isensor, osensor = Sensor("x"), Sensor("y")
        linear_model = LinearModel(["m", "b"], [isensor], [osensor])
        problem.add_forward_model("LinearModel", linear_model)

        # add experimental data
        np.random.seed(1)
        x_test = np.linspace(0.0, 1.0, 10)
        y_true = linear_model({isensor.name: x_test, "m": 2.5, "b": 1.7})[osensor.name]
        y_test = np.random.normal(loc=y_true, scale=0.5)
        problem.add_experiment(
            f"TestSeries_1",
            fwd_model_name="LinearModel",
            sensor_values={isensor.name: x_test, osensor.name: y_test},
        )

        # add the likelihood model
        problem.add_likelihood_model(
            GaussianLikelihoodModel(prms_def={"sigma": "std_model"}, sensors=osensor)
        )

        # here it is finally checked, that the rearrangement works
        pyro_solver = PyroSolver(problem, show_progress=False)
        computed_result = pyro_solver.dependency_dict
        expected_result = {"loc_m": [], "b": [], "sigma": [], "m": ["loc_m"]}
        self.assertEqual(computed_result, expected_result)
        # additionally check that the actual sampling works; note that the
        # target_accept_prob argument is stated here to check if it works to pass
        # additional keyword arguments to the run_mcmc method
        pyro_solver.run_mcmc(
            n_walkers=1, n_steps=100, n_initial_steps=10, target_accept_prob=0.8
        )

        # check the warning when the problem cannot be deep-copied
        problem.no_deepcopy_possible = (i for i in (1, 2, 3))
        PyroSolver(problem)  # this should result in a warning

    def test_evaluate_model_response(self):
        # prepare for checks
        p = InferenceProblem("TestProblem")
        p.add_parameter("a0", "model", prior=("normal", {"loc": 0, "scale": 1}))
        p.add_parameter("a1", "model", prior=("normal", {"loc": 0, "scale": 1}))
        p.add_parameter(
            "a2", "model", prior=("normal", {"loc": 0, "scale": 1}), domain="[0, 1]"
        )
        p.add_parameter("sigma", "likelihood", const=1.0)

        class FwdModel(ForwardModelBase):
            def response(self, inp):
                x = inp["x"]
                a0 = inp["a0"]
                a1 = inp["a1"]
                a2 = inp["a2"]
                return {"y": a0 + a1 * x + a2 * x ** 2}

        # add forward model
        fwd_model = FwdModel(["a0", "a1", "a2"], Sensor("x"), Sensor("y"))
        p.add_forward_model("FwdModel", fwd_model)

        # add experiment_names
        p.add_experiment(
            "Exp1", sensor_values={"x": 1, "y": 2}, fwd_model_name="FwdModel"
        )
        p.add_experiment(
            "Exp2", sensor_values={"x": 2, "y": 3}, fwd_model_name="FwdModel"
        )
        p.add_experiment(
            "Exp3", sensor_values={"x": [1, 2], "y": [1, 2]}, fwd_model_name="FwdModel"
        )

        # add the likelihood model
        p.add_likelihood_model(
            GaussianLikelihoodModel(
                prms_def={"sigma": "std_model"}, sensors=Sensor("y")
            )
        )

        # initialize the solver object
        pyro_solver = PyroSolver(p)

        # perform a check for all experiments
        a0_value, a1_value, a2_value = 1, 2, 3
        theta = th.tensor([a0_value, a1_value, a2_value])
        computed_result = pyro_solver.evaluate_model_response(theta)
        expected_result = {
            "Exp1": {"y": 6},
            "Exp2": {"y": 17},
            "Exp3": {"y": np.array([6, 17])},
        }

        # check for each item, because the last contains an array, which results in an
        # error is you do assertEqual on the whole thing
        self.assertEqual(computed_result["Exp1"], expected_result["Exp1"])
        self.assertEqual(computed_result["Exp2"], expected_result["Exp2"])
        self.assertEqual(
            computed_result["Exp3"]["y"][0], expected_result["Exp3"]["y"][0]
        )
        self.assertEqual(
            computed_result["Exp3"]["y"][1], expected_result["Exp3"]["y"][1]
        )

        # perform a check for a subset of experiments
        computed_result = pyro_solver.evaluate_model_response(
            theta, experiment_names="Exp3"
        )
        expected_result = {"Exp3": {"y": np.array([6, 17])}}
        self.assertEqual(
            computed_result["Exp3"]["y"][0], expected_result["Exp3"]["y"][0]
        )
        self.assertEqual(
            computed_result["Exp3"]["y"][1], expected_result["Exp3"]["y"][1]
        )
        # check the likelihood out of a parameter domain
        computed_result = pyro_solver.loglike(theta)
        self.assertEqual(computed_result, -np.infty)

    def test_likelihood_model_with_multiplicative_model_error(self):
        # prepare for checks
        p = InferenceProblem("TestProblem")
        p.add_parameter("a0", "model", prior=("normal", {"loc": 0, "scale": 1}))
        p.add_parameter("a1", "model", prior=("normal", {"loc": 0, "scale": 1}))
        p.add_parameter("a2", "model", prior=("normal", {"loc": 0, "scale": 1}))
        p.add_parameter("sigma", "likelihood", const=1.0)

        class FwdModel(ForwardModelBase):
            def response(self, inp):
                x = inp["x"]
                a0 = inp["a0"]
                a1 = inp["a1"]
                a2 = inp["a2"]
                return {"y": a0 + a1 * x + a2 * x ** 2}

        # add forward model
        fwd_model = FwdModel(["a0", "a1", "a2"], Sensor("x"), Sensor("y"))
        p.add_forward_model("M1", fwd_model)

        # add experiment_names
        p.add_experiment("Exp1", sensor_values={"x": 1, "y": 2}, fwd_model_name="M1")
        p.add_experiment("Exp2", sensor_values={"x": 2, "y": 3}, fwd_model_name="M1")

        # add the likelihood model
        p.add_likelihood_model(
            GaussianLikelihoodModel(
                prms_def={"sigma": "std_model"},
                sensors=Sensor("y"),
                multiplicative_model_error=True,
                additive_model_error=False,
            )
        )

        # initialize the solver object; this will raises an error since the
        # multiplicative model error is not implemented for pyro yet
        with self.assertRaises(NotImplementedError):
            PyroSolver(p)

    def test_likelihood_model_with_correlation(self):
        # prepare for checks
        p = InferenceProblem("TestProblem")
        p.add_parameter("a0", "model", prior=("normal", {"loc": 0, "scale": 1}))
        p.add_parameter("a1", "model", prior=("normal", {"loc": 0, "scale": 1}))
        p.add_parameter("a2", "model", prior=("normal", {"loc": 0, "scale": 1}))
        p.add_parameter("sigma", "likelihood", const=1.0)

        class FwdModel(ForwardModelBase):
            def response(self, inp):
                x = inp["x"]
                a0 = inp["a0"]
                a1 = inp["a1"]
                a2 = inp["a2"]
                return {"y": a0 + a1 * x + a2 * x ** 2}

        # add forward model
        fwd_model = FwdModel(["a0", "a1", "a2"], Sensor("x"), Sensor("y"))
        p.add_forward_model("M1", fwd_model)

        # add experiment_names
        p.add_experiment("Exp1", sensor_values={"x": 1, "y": 2}, fwd_model_name="M1")
        p.add_experiment("Exp2", sensor_values={"x": 2, "y": 3}, fwd_model_name="M1")

        # add the likelihood model
        p.add_likelihood_model(
            GaussianLikelihoodModel(
                prms_def={"sigma": "std_model"},
                sensors=Sensor("y"),
                correlation_variables="x",
            )
        )

        # initialize the solver object; this will raises an error since the
        # multiplicative model error is not implemented for pyro yet
        with self.assertRaises(NotImplementedError):
            PyroSolver(p)


if __name__ == "__main__":
    unittest.main()
