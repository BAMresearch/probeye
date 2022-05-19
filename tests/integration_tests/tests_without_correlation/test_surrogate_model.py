# standard library imports
import unittest
import time

# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports (problem definition)
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.surrogate_model import SurrogateModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel


class TestProblem(unittest.TestCase):
    def test_surrogate_model(self, plot: bool = False):
        """
        Demonstrates how to use a surrogate model. Still under discussion.

        Parameters
        ----------
        plot
            Triggers a plot of the data used for calibration.
        """

        # ============================================================================ #
        #                              Set numeric values                              #
        # ============================================================================ #

        # 'true' value of a, and its normal prior parameters
        m_true = 2.5
        mean_m = 2.0
        std_m = 1.0

        # 'true' value of b, and its normal prior parameters
        b_true = 1.7
        mean_b = 1.0
        std_b = 1.0

        # 'true' value of additive error sd, and its uniform prior parameters
        sigma = 0.5
        low_sigma = 0.0
        high_sigma = 0.8

        # the number of generated experiment_names and seed for random numbers
        n_tests = 50
        seed = 1

        # ============================================================================ #
        #                           Define the Forward Model                           #
        # ============================================================================ #

        class ExpensiveModel(ForwardModelBase):
            def interface(self):
                self.parameters = ["m", "b"]
                self.input_sensors = Sensor("x")
                self.output_sensors = Sensor("y", std_model="sigma")

            def response(self, inp: dict) -> dict:
                x = inp["x"]
                m = inp["m"]
                b = inp["b"]
                time.sleep(1.0)
                return {"y": m * x + b}

        # ============================================================================ #
        #                          Define the Surrogate Model                          #
        # ============================================================================ #

        class SurrogateModel(ExpensiveModel, SurrogateModelBase):
            """
            The inheritance from ExpensiveModel 'copies' the interface-method from
            ExpensiveModel (the surrogate model should have the same interface as the
            forward model). The inheritance from SurrogateModelBase is required to
            assign a forward model to the surrogate model, see surrogate_model.py.
            """

            def response(self, inp: dict) -> dict:
                x = inp["x"]
                m = inp["m"]
                b = inp["b"]
                return {"y": m * x + b}

        # ============================================================================ #
        #                         Define the Inference Problem                         #
        # ============================================================================ #

        # initialize the inverse problem with a useful name
        problem = InverseProblem("Using a surrogate model")

        # add all parameters to the problem
        problem.add_parameter(
            "m",
            "model",
            tex="$m$",
            info="Slope of the graph",
            prior=("normal", {"mean": mean_m, "std": std_m}),
        )
        problem.add_parameter(
            "b",
            "model",
            info="Intersection of graph with y-axis",
            tex="$b$",
            prior=("normal", {"mean": mean_b, "std": std_b}),
        )
        problem.add_parameter(
            "sigma",
            "likelihood",
            domain="(0, +oo)",
            tex=r"$\sigma$",
            info="Standard deviation, of zero-mean additive model error",
            prior=("uniform", {"low": low_sigma, "high": high_sigma}),
        )

        # add the surrogate model (which is essentially a forward model) to the problem;
        # whether or not the surrogate model gets the original forward model is not
        # finally decided on, but it might not be necessary
        forward_model = ExpensiveModel("ExpensiveModel")
        surrogate_model = SurrogateModel("FastModel", forward_model=forward_model)
        problem.add_forward_model(surrogate_model)

        # ============================================================================ #
        #                    Add test data to the Inference Problem                    #
        # ============================================================================ #

        # data-generation; normal likelihood with constant variance around each point
        np.random.seed(seed)
        x_test = np.linspace(0.0, 1.0, n_tests)
        y_true = surrogate_model.response(
            {surrogate_model.input_sensor.name: x_test, "m": m_true, "b": b_true}
        )[surrogate_model.output_sensor.name]
        y_test = np.random.normal(loc=y_true, scale=sigma)

        # add the experimental data
        problem.add_experiment(
            f"TestSeries_1",
            fwd_model_name="ExpensiveModel",
            sensor_values={
                forward_model.input_sensor.name: x_test,
                forward_model.output_sensor.name: y_test,
            },
        )

        # plot the true and noisy data
        if plot:
            plt.scatter(x_test, y_test, label="measured data", s=10, c="red", zorder=10)
            plt.plot(x_test, y_true, label="true", c="black")
            plt.xlabel(forward_model.input_sensor.name)
            plt.ylabel(forward_model.output_sensor.name)
            plt.legend()
            plt.tight_layout()
            plt.draw()  # does not stop execution

        # ============================================================================ #
        #                           Add likelihood model(s)                            #
        # ============================================================================ #

        # add the likelihood model to the problem
        problem.add_likelihood_model(
            GaussianLikelihoodModel(
                prms_def="sigma",
                experiment_name="TestSeries_1",
                model_error="additive",
                name="SimpleLikelihoodModel",
            )
        )

        # give problem overview
        problem.info()


if __name__ == "__main__":
    unittest.main()
