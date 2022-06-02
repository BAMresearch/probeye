# standard library
import unittest

# third party imports
import numpy as np

# local imports
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.distribution import Normal, Uniform
from probeye.definition.sensor import Sensor
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.inference.scipy.solver import ScipySolver


class TestProblem(unittest.TestCase):
    def test_scipy_solver(self):

        # define a simple linear model for this case
        class LinearModel(ForwardModelBase):
            def interface(self):
                self.parameters = ["m", "b"]
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y", std_model="sigma")

            def response(self, inp):
                return {"y": inp["m"] * inp["x"] + inp["b"]}

        # define parameters with with an uninformative prior
        problem = InverseProblem("Problem with uninformative prior")
        problem.add_parameter("m", "model")  # uninformative prior
        problem.add_parameter("b", "model", prior=Normal(mean=1, std=1))
        problem.add_parameter(
            "sigma",
            "likelihood",
            prior=Uniform(low=0.1, high=0.8),
        )

        # add forward model and likelihood model
        linear_model = LinearModel("LinearModel")
        problem.add_forward_model(linear_model)

        # add experimental data
        np.random.seed(1)
        x_test = np.linspace(0.0, 1.0, 10)
        y_true = linear_model({"x": x_test, "m": 2.5, "b": 1.7})["y"]
        y_test = np.random.normal(loc=y_true, scale=0.5)
        problem.add_experiment(
            f"TestSeries_1",
            fwd_model_name="LinearModel",
            sensor_values={"x": x_test, "y": y_test},
        )

        # add the likelihood model
        problem.add_likelihood_model(
            GaussianLikelihoodModel(prms_def="sigma", experiment_name="TestSeries_1")
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
        scipy_solver.run_max_likelihood(solver_options={"maxiter": 1000})

        # check the 'summarize_point_estimate_results' methods when ML is not successful
        no_success_results = scipy_solver.raw_results
        no_success_results.status = 1
        scipy_solver.summarize_point_estimate_results(
            no_success_results, true_values=None, x0_dict=x0_dict
        )

    def test_evaluate_model_response(self):
        # prepare for checks
        p = InverseProblem("TestProblem")
        p.add_parameter("a0", "model", prior=Normal(mean=0, std=1))
        p.add_parameter("a1", "model", prior=Normal(mean=0, std=1))
        p.add_parameter("a2", "model", prior=Normal(mean=0, std=1), domain="[0, 1]")
        p.add_parameter("sigma", "likelihood", const=1.0)

        class FwdModel(ForwardModelBase):
            def interface(self):
                self.parameters = ["a0", "a1", "a2"]
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y", std_model="sigma")

            def response(self, inp):
                x = inp["x"]
                a0 = inp["a0"]
                a1 = inp["a1"]
                a2 = inp["a2"]
                return {"y": a0 + a1 * x + a2 * x**2}

        # add forward and likelihood model
        fwd_model = FwdModel("FwdModel")
        p.add_forward_model(fwd_model)

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
            GaussianLikelihoodModel(prms_def="sigma", experiment_name="Exp1")
        )

        # initialize the solver object
        scipy_solver = ScipySolver(p)

        # perform a check for all experiments
        a0_value, a1_value, a2_value = 1, 2, 3
        theta = np.array([a0_value, a1_value, a2_value])
        fwd_model = scipy_solver.problem.forward_models["FwdModel"]

        comp_result, _ = scipy_solver.evaluate_model_response(theta, fwd_model, "Exp1")
        self.assertEqual(comp_result, np.array(6.0))

        comp_result, _ = scipy_solver.evaluate_model_response(theta, fwd_model, "Exp2")
        self.assertEqual(comp_result, np.array(17.0))

        comp_result, _ = scipy_solver.evaluate_model_response(theta, fwd_model, "Exp3")
        self.assertTrue(np.allclose(comp_result, np.array([6.0, 17.0])))

        # check the likelihood out of a parameter domain
        comp_result = scipy_solver.loglike(theta)
        self.assertEqual(comp_result, -np.infty)


if __name__ == "__main__":
    unittest.main()
