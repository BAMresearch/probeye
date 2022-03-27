"""
Linear regression example solved with maximum likelihood (without needing priors)
----------------------------------------------------------------------------------------
The model equation is y(x) = a * x + b with a, b being the model parameters, while the
likelihood model is based on a normal zero-mean additive model error distribution with
the standard deviation to infer. The problem is solved via maximum likelihood estimation
based on scipy.
"""

# standard library imports
import unittest

# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports (problem definition)
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# local imports (inference related)
from probeye.inference.scipy.solver import ScipySolver


class TestProblem(unittest.TestCase):
    def test_maximum_likelihood(self, plot: bool = False):
        """
        Integration test for the problem described at the top of this file.

        Parameters
        ----------
        plot
            If True, the problem's data is plotted. This is deactivated by default, so
            that the test does not stop until the generated plots are closed.
        """

        # ============================================================================ #
        #                              Set numeric values                              #
        # ============================================================================ #

        # 'true' value of m
        m_true = 2.5

        # 'true' value of b
        b_true = 1.7

        # 'true' value of additive model error sd
        sigma = 0.5

        # the number of generated experiment_names and seed for random numbers
        n_tests = 50
        seed = 1

        # ============================================================================ #
        #                           Define the Forward Model                           #
        # ============================================================================ #

        class LinearModel(ForwardModelBase):
            def definition(self):
                self.parameters = ["m", "b"]
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y")

            def response(self, inp: dict) -> dict:
                # this method *must* be provided by the user
                x = inp["x"]
                m = inp["m"]
                b = inp["b"]
                return {"y": m * x + b}

        # ============================================================================ #
        #                         Define the Inference Problem                         #
        # ============================================================================ #

        # initialize the inference problem with a useful name; note that the name will
        # only be stored as an attribute of the InferenceProblem and is not important
        # for the problem itself; can be useful when dealing with multiple problems
        problem = InferenceProblem("Max likelihood for linear regression")

        # add all parameters to the problem; the first argument states the parameter's
        # global name (here: 'm', 'b' and 'sigma'); the second argument defines the
        # parameter type (three options: 'model' for parameter's of the forward model,
        # 'prior' for prior parameters and 'likelihood' for parameters of the likelihood
        # model); the tex argument is states a tex-string for the parameter which is
        # only used for plotting
        problem.add_parameter("m", "model", tex="$m$")
        problem.add_parameter("b", "model", tex="$b$")
        problem.add_parameter("sigma", "likelihood", domain="(0, +oo)", tex=r"$\sigma$")

        # add the forward model to the problem
        linear_model = LinearModel()
        problem.add_forward_model("LinearModel", linear_model)

        # ============================================================================ #
        #                    Add test data to the Inference Problem                    #
        # ============================================================================ #

        # data-generation; normal likelihood with constant variance around each point
        np.random.seed(seed)
        x_test = np.linspace(0.0, 1.0, n_tests)
        y_true = linear_model.response(
            {linear_model.input_sensor.name: x_test, "m": m_true, "b": b_true}
        )[linear_model.output_sensor.name]
        y_test = np.random.normal(loc=y_true, scale=sigma)

        # add the experimental data
        problem.add_experiment(
            f"TestSeries_1",
            fwd_model_name="LinearModel",
            sensor_values={
                linear_model.input_sensor.name: x_test,
                linear_model.output_sensor.name: y_test,
            },
        )

        # plot the true and noisy data
        if plot:
            plt.scatter(x_test, y_test, label="measured data", s=10, c="red", zorder=10)
            plt.plot(x_test, y_true, label="true", c="black")
            plt.xlabel(linear_model.input_sensor.name)
            plt.ylabel(linear_model.output_sensor.name)
            plt.legend()
            plt.tight_layout()
            plt.draw()  # does not stop execution

        # ============================================================================ #
        #                              Add noise model(s)                              #
        # ============================================================================ #

        # add the noise model to the problem
        problem.add_likelihood_model(
            GaussianLikelihoodModel(prms_def={"sigma": "std_model"})
        )

        # give problem overview
        problem.info()

        # ============================================================================ #
        #                    Solve problem with inference engine(s)                    #
        # ============================================================================ #

        # this routine is imported from another script because it it used by all
        # integration tests in the same way; ref_values are used for plotting
        true_values = {"m": m_true, "b": b_true, "sigma": sigma}
        scipy_solver = ScipySolver(problem)
        scipy_solver.run_max_likelihood(true_values=true_values)


if __name__ == "__main__":
    unittest.main()
