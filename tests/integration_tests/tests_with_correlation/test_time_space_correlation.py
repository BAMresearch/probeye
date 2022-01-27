# standard library
import unittest

# third party imports
import numpy as np
import matplotlib.pyplot as plt
from tripy.base import MeasurementSpaceTimePoints
from tripy.utils import correlation_function

# local imports (problem definition)
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.definition.sensor import Sensor
from probeye.subroutines import len_or_one

# local imports (testing related)
from tests.integration_tests.subroutines import run_inference_engines


class TestProblem(unittest.TestCase):
    def test_time_space_correlation(
        self,
        n_steps: int = 200,
        n_initial_steps: int = 100,
        n_walkers: int = 20,
        plot: bool = False,
        show_progress: bool = False,
        run_scipy: bool = True,
        run_emcee: bool = False,
        run_torch: bool = False,
        run_dynesty: bool = False,
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
        """

        if run_torch:
            raise RuntimeError(
                "The pyro-solver is not available yet for forward models including "
                "correlations."
            )

        # ============================================================================ #
        #                              Set numeric values                              #
        # ============================================================================ #

        # length of the beam and weights of the cars
        L = 100.0  # [m]
        g = 9.81  # [m/s**2]
        F1 = 3000.0 * g  # [N]
        F2 = 5000.0 * g  # [N]
        F3 = 12000.0 * g  # [N]

        # positions of the two sensors
        x_sensor_1 = 15.0  # [m]
        x_sensor_2 = 85.0  # [m]

        # 'true' value of EI, and its normal prior parameters
        EI_true = 2.1e11 * 0.25  # [Nm^2]
        loc_EI = 0.9 * EI_true
        scale_EI = 0.25 * loc_EI

        # 'true' value of noise sd, and its uniform prior parameters
        sigma = 1e-3
        low_sigma = 1e-4
        high_sigma = 1e-2

        # 'true' value of spatial correlation length, and its uniform prior parameters
        l_corr_x = 10.0  # [m]
        low_l_corr_x = 1.0  # [m]
        high_l_corr_x = 25.0  # [m]

        # 'true' value of temporal correlation length, and its uniform prior parameters
        l_corr_t = 1.0  # [s]
        low_l_corr_t = 0.1  # [s]
        high_l_corr_t = 5.0  # [s]

        # settings for the data generation
        seed = 1

        # ============================================================================ #
        #                           Define the Forward Model                           #
        # ============================================================================ #

        class BeamModel(ForwardModelBase):
            @staticmethod
            def beam_deflect(x_sensor, x_load, L, F, EI):
                y = np.zeros(len_or_one(x_load))
                for i, x_load_i in enumerate(x_load):
                    if x_sensor <= x_load_i:
                        b = L - x_load_i
                        x = x_sensor
                    else:
                        b = x_load_i
                        x = L - x_sensor
                    y[i] = -(F * b * x) / (6 * L * EI) * (L ** 2 - b ** 2 - x ** 2)
                return y

            def response(self, inp: dict) -> dict:
                v = inp["v"]
                t = inp["t"]
                L = inp["L"]
                F = inp["F"]
                EI = inp["EI"]
                response = {}
                x_load = v * t
                for os in self.output_sensors:
                    response[os.name] = self.beam_deflect(os.x, x_load, L, F, EI)
                return response

        # ============================================================================ #
        #                             Additional functions                             #
        # ============================================================================ #

        def correlation_func_space(d):
            return correlation_function(d, correlation_length=l_corr_x)

        def correlation_func_time(d):
            return correlation_function(d, correlation_length=l_corr_t)

        # ============================================================================ #
        #                         Define the Inference Problem                         #
        # ============================================================================ #

        # initialize the inference problem with a useful name
        problem = InferenceProblem("Simply supported beam with time-space correlation")

        # add all parameters to the problem
        problem.add_parameter(
            "EI",
            "model",
            tex="$EI$",
            info="Bending stiffness of the beam [Nm^2]",
            prior=("normal", {"loc": loc_EI, "scale": scale_EI}),
        )
        problem.add_parameter(
            "L", "model", tex="$L$", info="Length of the beam [m]", const=L
        )
        problem.add_parameter(
            "sigma",
            "likelihood",
            tex=r"$\sigma$",
            info="Std. dev, of 0-mean noise model",
            prior=("uniform", {"low": low_sigma, "high": high_sigma}),
        )
        problem.add_parameter(
            "l_corr_x",
            "likelihood",
            tex=r"$l_\mathrm{corr, x}$",
            info="Spatial correlation length of correlation model",
            prior=("uniform", {"low": low_l_corr_x, "high": high_l_corr_x}),
        )
        problem.add_parameter(
            "l_corr_t",
            "likelihood",
            tex=r"$l_\mathrm{corr, t}$",
            info="Temporal correlation length of correlation model",
            prior=("uniform", {"low": low_l_corr_t, "high": high_l_corr_t}),
        )

        # add the forward model to the problem
        isensor_1 = Sensor("v")
        isensor_2 = Sensor("t")
        isensor_3 = Sensor("F")
        osensor_1 = Sensor("y1", x=x_sensor_1)
        osensor_2 = Sensor("y2", x=x_sensor_2)
        beam_model = BeamModel(
            ["L", "EI"], [isensor_1, isensor_2, isensor_3], [osensor_1, osensor_2]
        )
        problem.add_forward_model("BeamModel", beam_model)

        # ============================================================================ #
        #                    Add test data to the Inference Problem                    #
        # ============================================================================ #

        # for reproducible results
        np.random.seed(seed)

        # additional data on the three experiments
        n_vector = [50, 50, 50]
        v_vector = [2.5, 10.0, 5.0]  # [m/s]
        F_vector = [F1, F2, F3]
        c_vector = ["black", "red", "blue"]

        for j, (n, v, F, c) in enumerate(zip(n_vector, v_vector, F_vector, c_vector)):

            # compute the 'true' deflections which will serve as mean values
            t = np.linspace(0, L / v, n)
            inp_1 = {"v": v, "t": t, "L": L, "F": F, "EI": EI_true}
            mean_dict = beam_model.response(inp_1)
            mean = np.append(mean_dict[osensor_1.name], mean_dict[osensor_2.name])

            # compute the covariance matrix using tripy
            cov_compiler = MeasurementSpaceTimePoints()
            cov_compiler.add_measurement_space_points(
                coord_mx=[osensor_1.x, osensor_2.x],
                group="space",
                sensor_name=osensor_1.name,
                standard_deviation=sigma,
            )
            cov_compiler.add_measurement_time_points(coord_vec=t, group="time")
            cov_compiler.add_measurement_space_within_group_correlation(
                group="space", correlation_func=correlation_func_space
            )
            cov_compiler.add_measurement_time_within_group_correlation(
                group="time", correlation_func=correlation_func_time
            )
            cov = cov_compiler.compile_covariance_matrix()

            # generate the experimental data and add it to the problem
            y_test = np.random.multivariate_normal(mean=mean, cov=cov)
            problem.add_experiment(
                f"Test_{j + 1}",
                fwd_model_name="BeamModel",
                sensor_values={
                    isensor_1.name: v,
                    isensor_2.name: t,
                    isensor_3.name: F,
                    osensor_1.name: y_test[:n],
                    osensor_2.name: y_test[n:],
                },
            )

            # plot the data if requested
            if plot:

                # first sensor
                plt.plot(t, mean[:n], "--", label=f"y1 (mean, test {j + 1})", color=c)
                plt.plot(t, y_test[:n], "-o", label=f"y1 (sampled, test {j + 1})", c=c)

                # second sensor
                plt.plot(t, mean[n:], "-.", label=f"y2 (mean, test {j + 1})", color=c)
                plt.plot(t, y_test[n:], "-x", label=f"y2 (sampled, test {j + 1})", c=c)

        # finish and show the plot
        if plot:
            plt.xlabel("t [s]")
            plt.ylabel("deflection [m]")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.show()

        # ============================================================================ #
        #                              Add noise model(s)                              #
        # ============================================================================ #

        # add the log-likelihood model to the problem
        loglike = GaussianLikelihoodModel(
            [
                {"sigma": "std_model"},
                {"l_corr_x": "l_corr_space"},
                {"l_corr_t": "l_corr_time"},
            ],
            [osensor_1, osensor_2],
            additive_model_error=True,
            multiplicative_model_error=False,
            additive_measurement_error=False,
            experiment_names=["Test_1"],#, "Test_2", "Test_3"],
            correlation_variables="xt",
            correlation_model="exp",
        )
        problem.add_likelihood_model(loglike)

        # give problem overview
        problem.info()

        # ============================================================================ #
        #                    Solve problem with inference engine(s)                    #
        # ============================================================================ #

        # this routine is imported from another script because it it used by all
        # integration tests in the same way
        true_values = {
            "EI": EI_true,
            "sigma": sigma,
            "l_corr_x": l_corr_x,
            "l_corr_t": l_corr_t,
        }
        run_inference_engines(
            problem,
            true_values=true_values,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            n_walkers=n_walkers,
            plot=False,
            show_progress=show_progress,
            run_scipy=run_scipy,
            run_emcee=run_emcee,
            run_torch=run_torch,
            run_dynesty=run_dynesty,
        )


if __name__ == "__main__":
    unittest.main()
