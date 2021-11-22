"""
Simple linear regression example with two model and one noise parameter
--------------------------------------------------------------------------------
The model equation is y = a * x + b with a, b being the model parameters and the
noise model is a normal zero-mean distribution with the std. deviation to infer.
The problem is solved via sampling using emcee and pyro.
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
from probeye.definition.noise_model import NormalNoiseModel

# local imports (testing related)
from tests.integration_tests.subroutines import run_inference_engines


class TestProblem(unittest.TestCase):
    def test_linear_regression(
        self,
        n_steps=200,
        n_initial_steps=100,
        n_walkers=20,
        plot=False,
        show_progress=False,
        run_scipy=True,
        run_emcee=True,
        run_torch=True,
    ):
        """
        Integration test for the problem described at the top of this file.

        Parameters
        ----------
        n_steps : int, optional
            Number of steps (samples) to run. Note that the default number is
            rather low just so the test does not take too long.
        n_initial_steps : int, optional
            Number of steps for initial (burn-in) sampling.
        n_walkers : int, optional
            Number of walkers used by the estimator.
        plot : bool, optional
            If True, the data and the posterior distributions are plotted. This
            is deactivated by default, so that the test does not stop until the
            generated plots are closed.
        show_progress : bool, optional
            If True, progress-bars will be shown, if available.
        run_scipy : bool, optional
            If True, the problem is solved with scipy (maximum likelihood est).
            Otherwise, no maximum likelihood estimate is derived.
        run_emcee : bool, optional
            If True, the problem is solved with the emcee solver. Otherwise,
            the emcee solver will not be used.
        run_torch : bool, optional
            If True, the problem is solved with the pyro/torch_ solver.
            Otherwise, the pyro/torch_ solver will not be used.
        """

        # ==================================================================== #
        #                          Set numeric values                          #
        # ==================================================================== #

        # 'true' value of a, and its normal prior parameters
        a_true = 2.5
        loc_a = 2.0
        scale_a = 1.0

        # 'true' value of b, and its normal prior parameters
        b_true = 1.7
        loc_b = 1.0
        scale_b = 1.0

        # 'true' value of noise sd, and its uniform prior parameters
        sigma_noise = 0.5
        low_sigma = 0.1
        high_sigma = 0.8

        # the number of generated experiment_names and seed for random numbers
        n_tests = 50
        seed = 1

        # ==================================================================== #
        #                       Define the Forward Model                       #
        # ==================================================================== #

        class LinearModel(ForwardModelBase):
            def response(self, inp):
                # this method *must* be provided by the user
                x = inp["x"]
                m = inp["mb"][0]
                b = inp["mb"][1]
                response = {}
                for os in self.output_sensors:
                    response[os.name] = m * x + b
                return response

            def jacobian(self, inp):
                # this method *can* be provided by the user; if not provided
                # the jacobian will be approximated by finite differences
                x = inp["x"]  # vector
                one = np.ones(len(x))
                jacobian = {}
                for os in self.output_sensors:
                    # partial derivatives must only be stated for the model
                    # parameters; all other input must be flagged by None;
                    # note: partial derivatives must be given as column vectors
                    jacobian[os.name] = {
                        "x": None,  # x is not a model param.
                        "mb": np.array([x, one]).transpose(),
                    }
                return jacobian

        # ==================================================================== #
        #                     Define the Inference Problem                     #
        # ==================================================================== #

        # initialize the inference problem with a useful name; note that the
        # name will only be stored as an attribute of the InferenceProblem and
        # is not important for the problem itself; can be useful when dealing
        # with multiple problems
        problem = InferenceProblem("Linear regression with normal noise")

        # add all parameters to the problem; the first argument states the
        # parameter's global name (here: 'a', 'b' and 'sigma'); the second
        # argument defines the parameter type (three options: 'model' for
        # parameter's of the forward model, 'prior' for prior parameters and
        # 'noise' for parameters of the noise model); the 'info'-argument is a
        # short description string used for logging, and the tex-argument gives
        # a tex-string of the parameter used for plotting; finally, the prior-
        # argument specifies the parameter's prior; note that this definition
        # of a prior will result in the initialization of constant parameters of
        # type 'prior' in the background
        problem.add_parameter(
            "mb",
            "model",
            dim=2,
            tex="$mb$",
            info="Slope and intercept of the graph",
            prior=(
                "normal",
                {
                    "loc": np.array([loc_a, loc_b]),
                    "scale": np.array([[scale_a, 0], [0, scale_b]]),
                },
            ),
        )
        problem.add_parameter(
            "sigma",
            "noise",
            tex=r"$\sigma$",
            info="Std. dev, of 0-mean noise model",
            prior=("uniform", {"low": low_sigma, "high": high_sigma}),
        )

        # add the forward model to the problem; note that the first positional
        # argument [{'a': 'm'}, 'b'] passed to LinearModel defines the forward
        # model's parameters by name via a list with elements structured like
        # {<global parameter name>: <local parameter name>}; a global name is a
        # name introduced by problem.add_parameter, while a local name is a name
        # used in the response-method of the forward model class (see the class
        # LinearModel above); note that the use of the local parameter name 'm'
        # for the global parameter 'a' is added here only to highlight the
        # possibility of this feature; it is not necessary at all here; whenever
        # forward model's parameter has a similar local and global name (which
        # should be the case most of the times), one doesn't have to use the
        # verbose notation  {<global parameter name>: <local parameter name>}
        # but can instead just write the parameter's (global=local) name, like
        # it is done with the forward model's parameter 'b' below
        isensor = Sensor("x")
        osensor = Sensor("y")
        linear_model = LinearModel(["mb"], [isensor], [osensor])
        problem.add_forward_model("LinearModel", linear_model)

        # add the noise model to the problem
        problem.add_noise_model(
            NormalNoiseModel(prms_def={"sigma": "std"}, sensors=osensor)
        )

        # ==================================================================== #
        #                Add test data to the Inference Problem                #
        # ==================================================================== #

        # data-generation; normal noise with constant variance around each point
        np.random.seed(seed)
        x_test = np.linspace(0.0, 1.0, n_tests)
        y_true = linear_model.response({isensor.name: x_test, "mb": [a_true, b_true]})[
            osensor.name
        ]
        y_test = np.random.normal(loc=y_true, scale=sigma_noise)

        # add the experimental data
        problem.add_experiment(
            f"TestSeries_1",
            fwd_model_name="LinearModel",
            sensor_values={isensor.name: x_test, osensor.name: y_test},
        )

        # give problem overview
        problem.info()

        # plot the true and noisy data
        if plot:
            plt.scatter(x_test, y_test, label="measured data", s=10, c="red", zorder=10)
            plt.plot(x_test, y_true, label="true", c="black")
            plt.xlabel(isensor.name)
            plt.ylabel(osensor.name)
            plt.legend()
            plt.tight_layout()
            plt.draw()  # does not stop execution

        # ==================================================================== #
        #                Solve problem with inference engine(s)                #
        # ==================================================================== #

        # this routine is imported from another script because it it used by all
        # integration tests in the same way; ref_values are used for plotting
        true_values = {"mb": [a_true, b_true], "sigma": sigma_noise}
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
        )


if __name__ == "__main__":
    unittest.main()
