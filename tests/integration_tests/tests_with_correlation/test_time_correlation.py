"""
Estimation of gravitational constant assuming correlation in time
----------------------------------------------------------------------------------------
The experimental setup in this test is a ball that is vertically thrown upwards into the
air and which eventually falls down again. The trajectory of the ball (here, simply the
elevation of the ball) is tracked by two optical trackers. The data is generated with a
model that accounts for the air's friction-induced drag onto the ball (forward_process),
while the assumed forward model (TrajectoryModel) does not account for the friction. The
two trackers measure the same thing, i.e., the elevation of the ball, but they record
their data with different time discretizations. For the signals produced by the trackers
a correlation of the generated data in time is assumed. There are three parameters of
the data-generation model: the gravitational constant 'g', an additive model error std.
deviation 'sigma', and a time-correlation length 'l_corr'. These three parameters are
inferred using maximum likelihood estimation as well as sampling via emcee and dynesty.
"""

# standard library imports
import unittest
import os

# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports (problem definition)
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.interface.export_rdf import export_rdf

# local imports (testing related)
from tests.integration_tests.subroutines import run_inference_engines


class TestProblem(unittest.TestCase):
    def test_time_correlation(
        self,
        n_steps=200,
        n_initial_steps=100,
        n_walkers=20,
        plot=False,
        show_progress=False,
        run_scipy=True,
        run_emcee=True,
        run_torch=False,
        run_dynesty=True,
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
            If True, the problem is solved with the pyro/torch solver.
            Otherwise, the pyro/torch solver will not be used.
        run_dynesty
            If True, the problem is solved with the dynesty solver. Otherwise, the
            dynesty solver will not be used.
        """

        if run_torch:
            raise RuntimeError(
                "The pyro-solver is not available for inference problems including "
                "correlations yet."
            )

        # ============================================================================ #
        #                              Set numeric values                              #
        # ============================================================================ #

        # the acceleration due to gravity [m/s^2]
        g_true = 9.81
        loc_g = 10.0
        scale_g = 0.3

        # these are the additive error model's standard deviations of the elevation
        # measurements of the two trackers
        sd_error_tracker_1 = 0.1
        sd_error_tracker_2 = 0.2
        low_sigma = 0.01
        high_sigma = 0.5
        low_l_corr = 0.01
        high_l_corr = 1.0

        # settings for the data generation
        plot_data = False
        seed = 1

        # ============================================================================ #
        #                           Define the Forward Model                           #
        # ============================================================================ #

        class TrajectoryModel(ForwardModelBase):
            def definition(self):
                self.parameters = "g"
                self.input_sensors = [Sensor("t"), Sensor("v0")]
                self.output_sensors = Sensor("y")

            def response(self, inp):
                t = inp["t"]
                v0 = inp["v0"]
                g = inp["g"]
                return {"y": np.maximum((v0 - 0.5 * g * t) * t, 0.0)}

        # ============================================================================ #
        #                         Define the Inference Problem                         #
        # ============================================================================ #

        # initialize the inference problem with a useful name
        problem = InferenceProblem("Estimate gravitational acceleration")

        # add all parameters to the problem
        problem.add_parameter(
            "g",
            "model",
            domain="(0, +oo)",
            tex="$g$",
            info="Gravitational acceleration of earth",
            prior=("normal", {"loc": loc_g, "scale": scale_g}),
        )
        problem.add_parameter(
            "sigma",
            "likelihood",
            domain="(0, +oo)",
            tex=r"$\sigma$",
            info="Standard deviation, of zero-mean additive model error",
            prior=("uniform", {"low": low_sigma, "high": high_sigma}),
        )
        problem.add_parameter(
            "l_corr",
            "likelihood",
            domain="(0, +oo)",
            tex=r"$l_\mathrm{corr}$",
            info="Correlation length of correlation model",
            prior=("uniform", {"low": low_l_corr, "high": high_l_corr}),
        )
        problem.add_parameter("std_meas", "likelihood", const=0.1)

        # add the forward model to the problem
        trajectory_model = TrajectoryModel()
        problem.add_forward_model("TrajectoryModel", trajectory_model)

        # ============================================================================ #
        #                    Add test data to the Inference Problem                    #
        # ============================================================================ #

        def forward_process(t, v0, g=g_true, m=0.145, r=0.036, rho=1.29, cd=0.05):
            """
            Computes the trajectory of a ball that is thrown straight upwards
            from ground level (zero height), until it falls back on the ground.
            The model accounts for Newtonian air drag and - in this example -
            should represent the real (instead of the modeled) process. The
            formulas used below have been taken from 'Trajectory of a projectile
            with Newton drag': https://en.wikipedia.org/wiki/Projectile_motion.

            Parameters
            ----------
            t : float, int, numpy.ndarray
                The point(s) in time in seconds for which the position (i.e,
                the elevation) of the ball should be evaluated.
            v0 : float
                The initial speed of the ball at zero height in m/s.
            m : float, optional
                The mass of the ball in kg.
            r : float, optional
                The radius of the ball in m.
            g : float, optional
                The acceleration due to gravity in m/s^2.
            rho : float, optional
                The air's density in kg/m^3.
            cd : float, optional
                The drag coefficient of the ball (dimensionless).

            Returns
            -------
            y : float, numpy.ndarray
                The elevation of the ball [m] in the specified scenario at
                the given time(s) t.
            """

            # we need t in vector-format to proceed
            if type(t) in [float, int]:
                t_vector = np.array([t])
            elif type(t) is list:
                t_vector = np.array(t)
            else:
                t_vector = t

            # derived parameters
            mu = 0.5 * cd * (np.pi * r ** 2) * rho / m
            t_f = 1 / np.sqrt(g * mu)
            v_inf = np.sqrt(g / mu)

            # compute time and elevation when the ball reaches its highest point
            t_peak = t_f * np.arctan(v0 / v_inf)
            y_peak = -1 / mu * np.log(np.cos(t_peak / t_f))

            # compute the time when the ball falls back on the ground
            t_end = t_peak + t_f * np.arccosh(np.exp(mu * y_peak))

            # there are two different equations for the up-/downward direction,
            # so we have to do a check here, to see on which side we are
            y_vector = np.zeros(t_vector.shape)
            for i, t_i in enumerate(t_vector):
                if t_i <= t_peak:
                    y_vector[i] = y_peak + 1 / mu * np.log(np.cos((t_peak - t_i) / t_f))
                elif (t_peak < t_i) and (t_i < t_end):
                    y_vector[i] = y_peak - 1 / mu * np.log(
                        np.cosh((t_i - t_peak) / t_f)
                    )
                else:
                    y_vector[i] = 0.0

            # for t as a single number the should not be an array returned
            if y_vector.size == 1:
                y = float(y_vector)
            else:
                y = y_vector
            return y

        # set the seed for the noise generation
        np.random.seed(seed)

        # test data for first test with two trackers
        v0_test_1 = 20.0
        time_test_1_tracker_1 = np.arange(0, 5, 1.0)
        y_test_1_tracker_1 = forward_process(time_test_1_tracker_1, v0_test_1)
        y_test_1_tracker_1 = np.random.normal(
            loc=y_test_1_tracker_1, scale=sd_error_tracker_1
        )
        time_test_1_tracker_2 = np.arange(0, 5, 0.6)
        y_test_1_tracker_2 = forward_process(time_test_1_tracker_2, v0_test_1)
        y_test_1_tracker_2 = np.random.normal(
            loc=y_test_1_tracker_2, scale=sd_error_tracker_2
        )

        # test data for second test with two trackers
        v0_test_2 = 25.0
        time_test_2_tracker_1 = np.arange(0, 6, 0.7)
        y_test_2_tracker_1 = forward_process(time_test_2_tracker_1, v0_test_2)
        y_test_2_tracker_1 = np.random.normal(
            loc=y_test_2_tracker_1, scale=sd_error_tracker_1
        )
        time_test_2_tracker_2 = np.arange(0, 6, 0.8)
        y_test_2_tracker_2 = forward_process(time_test_2_tracker_2, v0_test_2)
        y_test_2_tracker_2 = np.random.normal(
            loc=y_test_2_tracker_2, scale=sd_error_tracker_2
        )

        # plot the generated data if requested
        if plot_data:
            plt.plot(
                time_test_1_tracker_1,
                y_test_1_tracker_1,
                label="Test 1, Tracker 1",
                marker="o",
            )
            plt.plot(
                time_test_1_tracker_2,
                y_test_1_tracker_2,
                label="Test 1, Tracker 2",
                marker="o",
            )
            plt.plot(
                time_test_2_tracker_1,
                y_test_2_tracker_1,
                label="Test 2, Tracker 1",
                marker="v",
            )
            plt.plot(
                time_test_2_tracker_2,
                y_test_2_tracker_2,
                label="Test 2, Tracker 2",
                marker="v",
            )
            plt.xlabel("time $t$ [s]")
            plt.ylabel("elevation $y$ [m]")
            plt.legend()
            plt.tight_layout()
            plt.show()

        # add the experimental data
        problem.add_experiment(
            f"Trajectory_1_Tracker_1",
            fwd_model_name="TrajectoryModel",
            sensor_values={
                trajectory_model.input_sensors[0].name: time_test_1_tracker_1,
                trajectory_model.input_sensors[1].name: v0_test_1,
                trajectory_model.output_sensor.name: y_test_1_tracker_1,
            },
        )
        problem.add_experiment(
            f"Trajectory_1_Tracker_2",
            fwd_model_name="TrajectoryModel",
            sensor_values={
                trajectory_model.input_sensors[0].name: time_test_1_tracker_2,
                trajectory_model.input_sensors[1].name: v0_test_1,
                trajectory_model.output_sensor.name: y_test_1_tracker_2,
            },
        )
        problem.add_experiment(
            f"Trajectory_2_Tracker_1",
            fwd_model_name="TrajectoryModel",
            sensor_values={
                trajectory_model.input_sensors[0].name: time_test_2_tracker_1,
                trajectory_model.input_sensors[1].name: v0_test_2,
                trajectory_model.output_sensor.name: y_test_2_tracker_1,
            },
        )
        problem.add_experiment(
            f"Trajectory_2_Tracker_2",
            fwd_model_name="TrajectoryModel",
            sensor_values={
                trajectory_model.input_sensors[0].name: time_test_2_tracker_2,
                trajectory_model.input_sensors[1].name: v0_test_2,
                trajectory_model.output_sensor.name: y_test_2_tracker_2,
            },
        )

        # ============================================================================ #
        #                           Add likelihood model(s)                            #
        # ============================================================================ #

        # the two experimental data sets (experiments) 'Trajectory_1_Tracker_1' and
        # 'Trajectory_1_Tracker_2' refer to the same event or trajectory, hence they are
        # referred by the same likelihood model
        likelihood_model_1 = GaussianLikelihoodModel(
            prms_def=[
                {"sigma": "std_model"},
                "l_corr",
            ],
            sensors=trajectory_model.output_sensor,
            correlation_variables="t",
            correlation_model="exp",
            experiment_names=["Trajectory_1_Tracker_1", "Trajectory_1_Tracker_2"],
            additive_model_error=True,
            multiplicative_model_error=False,
            additive_measurement_error=False,
        )
        problem.add_likelihood_model(likelihood_model_1)

        # the two experimental data sets (experiments) 'Trajectory_2_Tracker_1' and
        # 'Trajectory_2_Tracker_2' refer to the same event or trajectory, hence they are
        # referred by the same likelihood model
        likelihood_model_2 = GaussianLikelihoodModel(
            prms_def=[
                {"sigma": "std_model"},
                "l_corr",
            ],
            sensors=trajectory_model.output_sensor,
            correlation_variables="t",
            correlation_model="exp",
            experiment_names=["Trajectory_2_Tracker_1", "Trajectory_2_Tracker_2"],
            additive_model_error=True,
            multiplicative_model_error=False,
            additive_measurement_error=False,
        )
        problem.add_likelihood_model(likelihood_model_2)

        # give problem overview
        problem.info()

        # ============================================================================ #
        #                   Export the described problem to triples                    #
        # ============================================================================ #

        # create the knowledge graph and print it to file
        dir_path = os.path.dirname(__file__)
        ttl_file = os.path.join(dir_path, "../test_time_correlation.ttl")
        export_rdf(
            problem,
            ttl_file,
            include_explanations=True,
            write_array_data=True,
        )

        # ============================================================================ #
        #                    Solve problem with inference engine(s)                    #
        # ============================================================================ #

        # this routine is imported from another script because it it used by all
        # integration tests in the same way
        run_inference_engines(
            problem,
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
