"""
Inference problem with two forward models that share a common parameter
----------------------------------------------------------------------------------------
The first model equation is y = a * x + b with a, b being the model parameters and the
second model equation is y = alpha * x**2 + b where alpha is a new model parameter, and
b is the same model parameter as in the first model equation. Both forward models have
the same noise model with a normal zero-mean distribution where the standard deviation
is to be inferred. The problem is solved via max likelihood estimation and via sampling
using emcee, pyro and dynesty.
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
from probeye.definition.likelihood_model import NormalNoiseModel

# local imports (testing related)
from tests.integration_tests.subroutines import run_inference_engines


class TestProblem(unittest.TestCase):
    def test_two_models(
        self,
        n_steps: int = 200,
        n_initial_steps: int = 100,
        n_walkers: int = 20,
        plot: bool = False,
        show_progress: bool = False,
        run_scipy: bool = True,
        run_emcee: bool = True,
        run_torch: bool = True,
        run_dynesty: bool = True,
    ):
        """
        Integration test for the problem described at the top of this file.

        Parameters
        ----------
        n_steps
            Number of steps (samples) to run. Note that the default number is rather low
            just so the test does not take too long.
        n_initial_steps
            Number of steps for initial (burn-in) sampling.
        n_walkers
            Number of walkers used by the estimator.
        plot
            If True, the data and the posterior distributions are plotted. This is
            deactivated by default, so that the test does not stop until the generated
            plots are closed.
        show_progress
            If True, progress-bars will be shown, if available.
        run_scipy
            If True, the problem is solved with scipy (maximum likelihood est).
            Otherwise, no maximum likelihood estimate is derived.
        run_emcee
            If True, the problem is solved with the emcee solver. Otherwise, the emcee
            solver will not be used.
        run_torch
            If True, the problem is solved with the pyro/torch_ solver. Otherwise, the
            pyro/torch_ solver will not be used.
        run_dynesty
            If True, the problem is solved with the dynesty solver. Otherwise, the
            dynesty solver will not be used.
        """

        # ============================================================================ #
        #                              Set numeric values                              #
        # ============================================================================ #

        # 'true' value of a, and its normal prior parameters
        a_true = 2.5
        loc_a = 2.0
        scale_a = 1.0

        # 'true' value of b, and its normal prior parameters
        b_true = 1.7
        loc_b = 1.0
        scale_b = 1.0

        # 'true' value of alpha, and its normal prior parameters
        alpha_true = 0.7
        loc_alpha = 2.0
        scale_alpha = 1.0

        # 'true' value of sigma, and its normal prior parameters
        sigma_true = 0.15
        low_sigma = 0.1
        high_sigma = 2.0

        # the number of generated experiment_names and seed for random numbers
        n_tests = 100
        seed = 1

        # ============================================================================ #
        #                          Define the Forward Models                           #
        # ============================================================================ #

        class LinearModel(ForwardModelBase):
            def response(self, inp: dict) -> dict:
                x = inp["x"]
                a = inp["a"]
                b = inp["b"]
                response = {}
                for os in self.output_sensors:
                    response[os.name] = a * x + b
                return response

        class QuadraticModel(ForwardModelBase):
            def response(self, inp: dict) -> dict:
                x = inp["x"]
                alpha = inp["alpha"]
                beta = inp["beta"]
                response = {}
                for os in self.output_sensors:
                    response[os.name] = alpha * x ** 2 + beta
                return response

        # ============================================================================ #
        #                         Define the Inference Problem                         #
        # ============================================================================ #

        # initialize the inference problem with a useful name
        problem = InferenceProblem("Two models with shared parameter and normal noise")

        # add all parameters to the problem
        problem.add_parameter(
            "a",
            "model",
            info="Slope of the graph in linear model",
            tex="$a$ (linear)",
            prior=("normal", {"loc": loc_a, "scale": scale_a}),
        )
        problem.add_parameter(
            "alpha",
            "model",
            info="Factor of quadratic term",
            tex=r"$\alpha$ (quad.)",
            prior=("normal", {"loc": loc_alpha, "scale": scale_alpha}),
        )
        problem.add_parameter(
            "b",
            "model",
            info="Intersection of graph with y-axis",
            tex="$b$ (shared)",
            prior=("normal", {"loc": loc_b, "scale": scale_b}),
        )
        problem.add_parameter(
            "sigma",
            "noise",
            tex=r"$\sigma$ (noise)",
            info="Std. deviation of zero-mean noise model",
            prior=("uniform", {"low": low_sigma, "high": high_sigma}),
        )

        # add the forward model to the problem
        isensor = Sensor("x")
        osensor_linear = Sensor("y_linear")
        osensor_quadratic = Sensor("y_quadratic")
        linear_model = LinearModel(["a", "b"], [isensor], [osensor_linear])
        problem.add_forward_model("LinearModel", linear_model)
        quadratic_model = QuadraticModel(
            ["alpha", {"b": "beta"}], [isensor], [osensor_quadratic]
        )
        problem.add_forward_model("QuadraticModel", quadratic_model)

        # ============================================================================ #
        #                    Add test data to the Inference Problem                    #
        # ============================================================================ #

        # data-generation; normal noise with constant variance around each point
        np.random.seed(seed)
        x_test = np.linspace(0.0, 1.0, n_tests)
        y_linear_true = linear_model({isensor.name: x_test, "a": a_true, "b": b_true})[
            osensor_linear.name
        ]
        y_test_linear = np.random.normal(loc=y_linear_true, scale=sigma_true)
        y_quadratic_true = quadratic_model(
            {isensor.name: x_test, "alpha": alpha_true, "beta": b_true}
        )[osensor_quadratic.name]
        y_test_quadratic = np.random.normal(loc=y_quadratic_true, scale=sigma_true)

        # add the experimental data
        problem.add_experiment(
            f"TestSeries_linear",
            sensor_values={isensor.name: x_test, osensor_linear.name: y_test_linear},
            fwd_model_name="LinearModel",
        )
        problem.add_experiment(
            f"TestSeries_quadratic",
            sensor_values={
                isensor.name: x_test,
                osensor_quadratic.name: y_test_quadratic,
            },
            fwd_model_name="QuadraticModel",
        )

        # plot the true and noisy data
        if plot:
            plt.scatter(
                x_test,
                y_test_linear,
                label="measured data (linear)",
                s=10,
                c="red",
                zorder=10,
            )
            plt.plot(x_test, y_linear_true, label="true (linear)", c="black")
            plt.scatter(
                x_test,
                y_test_quadratic,
                s=10,
                c="orange",
                zorder=10,
                label="measured data (quadratic)",
            )
            plt.plot(x_test, y_quadratic_true, label="true (quadratic)", c="blue")
            plt.xlabel(isensor.name)
            plt.ylabel(f"{osensor_linear.name}, {osensor_quadratic.name}")
            plt.legend()
            plt.tight_layout()
            plt.draw()  # does not stop execution

        # ============================================================================ #
        #                              Add noise model(s)                              #
        # ============================================================================ #

        # add the noise models to the problem
        problem.add_likelihood_model(
            NormalNoiseModel(prms_def={"sigma": "std"}, sensors=osensor_linear)
        )
        problem.add_likelihood_model(
            NormalNoiseModel(prms_def={"sigma": "std"}, sensors=osensor_quadratic)
        )

        # give problem overview
        problem.info()

        # ============================================================================ #
        #                    Solve problem with inference engine(s)                    #
        # ============================================================================ #

        # this routine is imported from another script because it it used by all
        # integration tests in the same way
        true_values = {
            "a": a_true,
            "alpha": alpha_true,
            "b": b_true,
            "sigma": sigma_true,
        }
        run_inference_engines(
            problem,
            true_values=true_values,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            n_walkers=n_walkers,
            plot=plot,
            show_progress=show_progress,
            run_scipy=run_scipy,
            run_emcee=run_emcee,
            run_torch=run_torch,
            run_dynesty=run_dynesty,
        )


if __name__ == "__main__":
    unittest.main()
