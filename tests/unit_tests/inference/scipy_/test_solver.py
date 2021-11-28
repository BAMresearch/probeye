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
from probeye.inference.scipy_.solver import ScipySolver


class TestProblem(unittest.TestCase):
    def test_scipy_solver(self):

        # define a simple linear model for this case
        class LinearModel(ForwardModelBase):
            def response(self, inp):
                return {"y": inp["m"] * inp["x"] + inp["b"]}

        # define parameters with with an uninformative prior
        problem = InferenceProblem("Problem with uninformative prior")
        problem.add_parameter("m", "model")  # uninformative prior
        problem.add_parameter(
            "b", "model", prior=("normal", {"loc": 1.0, "scale": 1.0})
        )
        problem.add_parameter(
            "sigma", "noise", prior=("uniform", {"low": 0.1, "high": 0.8})
        )

        # add forward model and noise model
        isensor, osensor = Sensor("x"), Sensor("y")
        linear_model = LinearModel(["m", "b"], [isensor], [osensor])
        problem.add_forward_model("LinearModel", linear_model)
        problem.add_noise_model(
            NormalNoiseModel(prms_def={"sigma": "std"}, sensors=osensor)
        )

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

        # test the get_start_values method for given x0_dict
        scipy_solver = ScipySolver(problem)
        x0, x0_dict = scipy_solver.get_start_values(
            x0_dict={"m": 2.5, "b": 1.5, "sigma": 0.6}
        )
        self.assertTrue(np.allclose(x0, np.array([2.5, 1.5, 0.6])))
        self.assertEqual(x0_dict, {"m": 2.5, "b": 1.5, "sigma": 0.6})

        # test the get_start_values method for automatic derivation
        scipy_solver = ScipySolver(problem)
        x0, x0_dict = scipy_solver.get_start_values(x0_prior="mean")
        self.assertTrue(np.allclose(x0, np.array([1.0, 1.0, 0.45])))
        expected_x0_dict = {"m": 1.0, "b": 1.0, "sigma": 0.45}
        for key, val in x0_dict.items():
            self.assertAlmostEqual(val, expected_x0_dict[key])

        # check that the solver can be run now; note that the additional solver option
        # is a default value, and only provided to check that this argument-pipeline
        # works
        scipy_solver.run_max_likelihood(solver_options={"jac": None})

        # check the warning when the problem cannot be deep-copied
        problem.no_deepcopy_possible = (i for i in (1, 2, 3))
        ScipySolver(problem)  # this should result in a warning

    def test_evaluate_model_response(self):
        # prepare for checks
        p = InferenceProblem("TestProblem")
        p.add_parameter("a0", "model", prior=("normal", {"loc": 0, "scale": 1}))
        p.add_parameter("a1", "model", prior=("normal", {"loc": 0, "scale": 1}))
        p.add_parameter("a2", "model", prior=("normal", {"loc": 0, "scale": 1}))
        p.add_parameter("sigma", "noise", const=1.0)

        class FwdModel(ForwardModelBase):
            def response(self, inp):
                x = inp["x"]
                a0 = inp["a0"]
                a1 = inp["a1"]
                a2 = inp["a2"]
                return {"y": a0 + a1 * x + a2 * x ** 2}

        # add forward and noise model
        fwd_model = FwdModel(["a0", "a1", "a2"], Sensor("x"), Sensor("y"))
        p.add_forward_model("FwdModel", fwd_model)
        p.add_noise_model(
            NormalNoiseModel(prms_def={"sigma": "std"}, sensors=Sensor("y"))
        )

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

        # initialize the solver object
        pyro_solver = ScipySolver(p)

        # perform a check for all experiments
        a0_value, a1_value, a2_value = 1, 2, 3
        theta = np.array([a0_value, a1_value, a2_value])
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


if __name__ == "__main__":
    unittest.main()
