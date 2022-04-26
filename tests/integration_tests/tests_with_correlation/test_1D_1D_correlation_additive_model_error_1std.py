"""
   Bending stiffness of simply supported beam assuming time-space (1D-1D) correlation
----------------------------------------------------------------------------------------
      ---> Additive model prediction error (common std. dev. across sensors) <---
----------------------------------------------------------------------------------------
A bridge (modeled as a simply supported beam) is equipped at two positions with a
deflection sensor. Both sensors record a time series of deflection while cars with
different weights and velocities cross the bridge (these are the different experiments).
Correlation is assumed in both space and time (separable). The goal of the inference is
to estimate the bridge's bending stiffness 'EI'. Next to 'EI' there are three other
parameters to infer: the additive model error standard deviation 'sigma', the temporal
correlation length 'l_corr_t' and the spatial correlation length l_corr_x. Hence, four
parameters in total, all of which are inferred in this example using a maximum
likelihood estimation.
"""

# standard library
import unittest

# third party imports
import numpy as np
import matplotlib.pyplot as plt
from tripy.base import MeasurementSpaceTimePoints
from tripy.utils import correlation_function

# local imports (problem definition)
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.definition.sensor import Sensor
from probeye.subroutines import len_or_one

# local imports (testing related)
from tests.integration_tests.subroutines import run_inference_engines


class TestProblem(unittest.TestCase):
    def test_1D_1D_correlation_additive_model_error_1std(
        self,
        n_steps: int = 200,
        n_initial_steps: int = 100,
        n_walkers: int = 20,
        plot: bool = False,
        show_progress: bool = False,
        run_scipy: bool = True,
        run_emcee: bool = False,  # intentionally False for faster test-runs
        run_dynesty: bool = False,  # intentionally False for faster test-runs
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
        run_dynesty
            If True, the problem is solved with the dynesty solver. Otherwise, the
            dynesty solver will not be used.
        """

        # ============================================================================ #
        #                              Set numeric values                              #
        # ============================================================================ #

        # definition of the experiments (a car driving over the bridge)
        experiments_def = {
            "Experiment_1": {
                "car_mass_kg": 3000.0,
                "car_speed_m/s": 2.5,
                "dt_sensors_sec": 0.4,
                "plot_color": "black",
            },
            "Experiment_2": {
                "car_mass_kg": 5000.0,
                "car_speed_m/s": 10,
                "dt_sensors_sec": 0.5,
                "plot_color": "red",
            },
            "Experiment_3": {
                "car_mass_kg": 10000.0,
                "car_speed_m/s": 5.0,
                "dt_sensors_sec": 0.6,
                "plot_color": "blue",
            },
        }

        # length of the beam and weights of the cars
        L = 100.0  # [m]
        g = 9.81  # [m/s**2]

        # positions of the sensors
        x_sensor_1 = 30.0  # [m]
        x_sensor_2 = 35.0  # [m]

        # 'true' value of EI, and its normal prior parameters
        EI_true = 2.1 * 0.25  # [Nm^2 / 1e11]
        mean_EI = 0.9 * EI_true
        std_EI = 0.25 * mean_EI

        # 'true' value of add. model error sd, and its uniform prior parameters
        sigma = 1e-3
        low_sigma = 0.0
        high_sigma = 1e-2

        # 'true' value of spatial correlation length, and its uniform prior parameters
        l_corr_x = 10.0  # [m]
        low_l_corr_x = 0.0  # [m]
        high_l_corr_x = 25.0  # [m]

        # 'true' value of temporal correlation length, and its uniform prior parameters
        l_corr_t = 1.0  # [s]
        low_l_corr_t = 0.0  # [s]
        high_l_corr_t = 5.0  # [s]

        # settings for the data generation
        ns = 2  # two sensors in this example
        seed = 1

        # ============================================================================ #
        #                           Define the Forward Model                           #
        # ============================================================================ #

        class BeamModel(ForwardModelBase):
            def interface(self):
                self.parameters = ["L", "EI"]
                self.input_sensors = [Sensor("v"), Sensor("t"), Sensor("F")]
                self.output_sensors = [
                    Sensor(
                        name="y1",
                        x=x_sensor_1,
                        std_model="sigma",
                        correlated_in={"x": "l_corr_x", "t": "l_corr_t"},
                    ),
                    Sensor(
                        name="y2",
                        x=x_sensor_2,
                        std_model="sigma",
                        correlated_in={"x": "l_corr_x", "t": "l_corr_t"},
                    ),
                ]

            @staticmethod
            def beam_deflect(x_sensor, x_load, L_in, F_in, EI_in):
                """Convenience method used by self.response during a for-loop."""
                y = np.zeros(len_or_one(x_load))
                for ii, x_load_i in enumerate(x_load):
                    if x_sensor <= x_load_i:
                        b = L - x_load_i
                        x = x_sensor
                    else:
                        b = x_load_i
                        x = L_in - x_sensor
                    y[ii] = (
                        -(F_in * b * x)
                        / (6 * L * EI_in)
                        * (L_in**2 - b**2 - x**2)
                    )
                return y

            def response(self, inp: dict) -> dict:
                v_in = inp["v"]
                t_in = inp["t"]
                L_in = inp["L"]
                F_in = inp["F"]
                EI_in = inp["EI"] * 1e11  # de-normalization
                response = {}
                x_load = v_in * t_in
                for os in self.output_sensors:
                    response[os.name] = self.beam_deflect(
                        os.x, x_load, L_in, F_in, EI_in
                    )
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

        # initialize the inverse problem with a useful name
        problem = InverseProblem(
            "Simply supported beam with time-space correlation (1D-1D, AME, 1std)",
            log_level="DEBUG",
        )

        # add all parameters to the problem
        problem.add_parameter(
            "EI",
            "model",
            domain="(0, +oo)",
            tex="$EI$",
            info="Bending stiffness of the beam [Nm^2]",
            prior=("normal", {"mean": mean_EI, "std": std_EI}),
        )
        problem.add_parameter(
            "L", "model", tex="$L$", info="Length of the beam [m]", const=L
        )
        problem.add_parameter(
            "sigma",
            "likelihood",
            domain="(0, +oo)",
            tex=r"$\sigma$",
            info="Standard deviation of zero-mean model error for both sensors",
            prior=("uniform", {"low": low_sigma, "high": high_sigma}),
        )
        problem.add_parameter(
            "l_corr_x",
            "likelihood",
            domain="(0, +oo)",
            tex=r"$l_\mathrm{corr,x}$",
            info="Spatial correlation length of correlation model",
            prior=("uniform", {"low": low_l_corr_x, "high": high_l_corr_x}),
        )
        problem.add_parameter(
            "l_corr_t",
            "likelihood",
            domain="(0, +oo)",
            tex=r"$l_\mathrm{corr,t}$",
            info="Temporal correlation length of correlation model",
            prior=("uniform", {"low": low_l_corr_t, "high": high_l_corr_t}),
        )

        # add the forward model to the problem
        beam_model = BeamModel("BeamModel")
        problem.add_forward_model(beam_model)

        # ============================================================================ #
        #                    Add test data to the Inference Problem                    #
        # ============================================================================ #

        # for reproducible results
        np.random.seed(seed)

        # add an experiment to the problem for for each item in experiments_def
        for exp_name, exp_dict in experiments_def.items():
            dt = exp_dict["dt_sensors_sec"]
            v = exp_dict["car_speed_m/s"]
            F = exp_dict["car_mass_kg"] * g  # type: ignore
            c = exp_dict["plot_color"]

            # compute the 'true' deflections for each sensor which will serve as mean
            # values; note that the values are concatenated to a long vector
            t = np.arange(0, L / v, dt)  # type: ignore
            nt = len(t)
            inp_1 = {"v": v, "t": t, "L": L, "F": F, "EI": EI_true}
            mean_dict = beam_model.response(inp_1)
            mean = np.zeros(ns * nt)
            for i, mean_vector in enumerate([*mean_dict.values()]):
                mean[i::ns] = mean_vector

            # compute the covariance matrix using tripy
            cov_compiler = MeasurementSpaceTimePoints()
            cov_compiler.add_measurement_space_points(
                coord_mx=[
                    beam_model.output_sensors[0].x,
                    beam_model.output_sensors[1].x,
                ],
                standard_deviation=np.array([sigma, sigma]),
                group="space",
            )
            cov_compiler.add_measurement_time_points(coord_vec=t, group="time")
            cov_compiler.add_measurement_space_within_group_correlation(
                group="space", correlation_func=correlation_func_space
            )
            cov_compiler.add_measurement_time_within_group_correlation(
                group="time", correlation_func=correlation_func_time
            )
            # note here that the rows/columns have the reference order:
            # y1(t1), y2(t1), y3(t1), ..., y1(t2), y2(t2), y3(t2), ....
            cov = cov_compiler.compile_covariance_matrix()

            # generate the experimental data and add it to the problem
            y_test = np.random.multivariate_normal(mean=mean, cov=cov)
            y1 = y_test[0::ns]
            y2 = y_test[1::ns]

            # finally, add the experiment to the problem
            problem.add_experiment(
                exp_name,
                fwd_model_name="BeamModel",
                sensor_values={
                    beam_model.input_sensors[0].name: v,
                    beam_model.input_sensors[1].name: t,
                    beam_model.input_sensors[2].name: F,
                    beam_model.output_sensors[0].name: y1,
                    beam_model.output_sensors[1].name: y2,
                },
            )

            # plot the data if requested
            if plot:

                # first sensor
                plt.plot(t, mean[0::ns], "-", label=f"y1 (true, {exp_name})", color=c)
                plt.scatter(t, y1, marker="o", label=f"y1 (sampled, {exp_name})", c=c)

                # second sensor
                plt.plot(t, mean[1::ns], "--", label=f"y2 (true, {exp_name})", color=c)
                plt.scatter(t, y2, marker="x", label=f"y2 (sampled, {exp_name})", c=c)

        # finish and show the plot
        if plot:
            plt.xlabel("t [s]")
            plt.ylabel("deflection [m]")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.show()

        # ============================================================================ #
        #                           Add likelihood model(s)                            #
        # ============================================================================ #

        # each likelihood model is assigned exactly one experiment
        for exp_name in problem.experiments.keys():
            loglike = GaussianLikelihoodModel(
                ["sigma", "l_corr_x", "l_corr_t"],
                experiment_name=exp_name,
                model_error="additive",
                correlation_variables=["x", "t"],
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
            plot=plot,
            show_progress=show_progress,
            run_scipy=run_scipy,
            run_emcee=run_emcee,
            run_dynesty=run_dynesty,
        )


if __name__ == "__main__":
    unittest.main()
