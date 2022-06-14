"""
Simple bridge model with time-space correlation
===============================================
A bridge (modeled as a simply supported beam) is equipped at multiple positions with a
deflection sensor. All sensors record a time series of deflection while cars with
different weights and velocities cross the bridge. Correlation is assumed in both space
and time. The goal of the inference is to estimate the bridge's bending stiffness 'EI'.
Next to 'EI' there are three other parameters to infer: the additive model error std.
deviation 'sigma', the temporal correlation length 'l_corr_t' and the spatial corr.
length l_corr_x. Hence, four parameters in total, all of which are inferred in this
example using maximum likelihood estimation as well as sampling via emcee.
"""

# %%
# First, let's import the required functions and classes for this example.

# third party imports
import numpy as np
import matplotlib.pyplot as plt
from tripy.base import MeasurementSpaceTimePoints
from tripy.utils import correlation_function

# local imports (problem definition)
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.distribution import Normal, Uniform
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.definition.correlation_model import ExpModel
from probeye.definition.sensor import Sensor
from probeye.subroutines import len_or_one
from probeye.subroutines import HiddenPrints

# local imports (problem solving)
from probeye.inference.scipy.solver import ScipySolver
from probeye.inference.emcee.solver import EmceeSolver

# local imports (inference data post-processing)
from probeye.postprocessing.sampling import create_pair_plot
from probeye.postprocessing.sampling import create_posterior_plot
from probeye.postprocessing.sampling import create_trace_plot

# %%
# The bridge that is considered in this example should have a length of 'L_bridge',
# while the positions of the 'ns' deflection sensors it is equipped with are given by
# the list 'x_sensors'. Note that the bridge is modeled as a 1D bridge, where the
# coordinates of the 'ns' sensors are measured from the side of the bridge where the
# cars enter the bridge. All length measurements are stated in meters. For the following
# computations we also need the gravitational constant, which is also given below.
# Finally, we need a true value for the bridge's bending stiffness 'EI_true'.

# relevant length measurements; feel free to add more sensors by adding the respective
# positions in the x_sensors-list
L_bridge = 100.0  # [m]
x_sensors = [49, 51, 55]

# other relevant constants
g = 9.81  # [m/s**2]

# 'true' value of EI
EI_true = 1.0  # [10^10 Nm^2]

# %%
# We will now begin with artificially generating our data. To that end, let's define
# three experiments, that is three scenarios where a car with a certain mass is crossing
# the bridge at a certain velocity. Feel free to add more experiments here.

# definition of the experiments
experiments_def = {
    "Experiment_1": {
        "car_mass_kg": 3000.0,
        "car_speed_m/s": 2.5,
        "plot_color": "black",
    },
    "Experiment_2": {
        "car_mass_kg": 5000.0,
        "car_speed_m/s": 10,
        "plot_color": "blue",
    },
    "Experiment_3": {
        "car_mass_kg": 10000.0,
        "car_speed_m/s": 5.0,
        "plot_color": "red",
    },
}

# %%
# Let's now define the parameters required to define the time-space correlated noise
# that we need to generate our synthetic test data.

# 'true' value of noise sd, and its uniform prior parameters
sigma = 0.01
low_sigma = 0.0
high_sigma = 0.2

# 'true' value of spatial correlation length, and its uniform prior parameters
l_corr_x = 10.0  # [m]
low_l_corr_x = 1.0  # [m]
high_l_corr_x = 25.0  # [m]

# 'true' value of temporal correlation length, and its uniform prior parameters
l_corr_t = 1.0  # [s]
low_l_corr_t = 0.0  # [s]
high_l_corr_t = 5.0  # [s]

# settings for the data generation
ns = len(x_sensors)
dt = 0.5  # [s]
seed = 1

# %%
# We'll continue with defining the forward model. Note however, that we are just
# defining it as a class, but we do not add it to the problem yet. The reason for
# defining it now is that we will use it to artificially generate our test data.


class BridgeModel(ForwardModelBase):
    def interface(self):
        self.parameters = ["L", "EI"]
        self.input_sensors = [Sensor("v"), Sensor("t"), Sensor("F")]
        self.output_sensors = []
        for i, x_i in enumerate(x_sensors):
            self.output_sensors.append(
                Sensor(name=f"y{i + 1}", x=x_i, std_model="sigma")
            )

    @staticmethod
    def beam_deflect(x_sensor, x_load, L_in, F_in, EI_in):
        """Convenience method used by self.response during a for-loop."""
        y = np.zeros(len_or_one(x_load))
        for i, x_load_i in enumerate(x_load):
            if x_sensor <= x_load_i:
                b = L_in - x_load_i
                x = x_sensor
            else:
                b = x_load_i
                x = L_in - x_sensor
            y[i] = -(F_in * b * x) / (6 * L_in * EI_in) * (L_in**2 - b**2 - x**2)
        return y

    def response(self, inp: dict) -> dict:
        v_in = inp["v"]
        t_in = inp["t"]
        L_in = inp["L"]
        F_in = inp["F"]
        EI_in = inp["EI"] * 1e10
        response = {}
        x_load = v_in * t_in
        for os in self.output_sensors:
            response[os.name] = self.beam_deflect(os.x, x_load, L_in, F_in, EI_in)
        return response


# %%
# Next to the forward model, we need to define two correlation functions in order to be
# able to generate  the data. Please note that this is just required for the data
# generation. It is not required for setting up the probeye framework. If we had real
# data, we would not have to do this.


def correlation_func_space(d):
    return correlation_function(d, correlation_length=l_corr_x)


def correlation_func_time(d):
    return correlation_function(d, correlation_length=l_corr_t)


# %%
# At this point, we have prepared all ingredients for generating our synthetic data.
# This code block is a bit longer, since we have to account for the non-trivial
# time-space correlation structure (we use tripy for this).

# initialize the bridge model
bridge_model = BridgeModel("BridgeModel")

# for reproducible results
np.random.seed(seed)

# create and plot data for each experiment
data_dict = {}  # type: dict
for j, (exp_name, exp_dict) in enumerate(experiments_def.items()):

    # load experimental setting
    v = exp_dict["car_speed_m/s"]
    F = exp_dict["car_mass_kg"] * g  # type: ignore

    # compute the 'true' deflections for each sensor which will serve as mean
    # values; note that the values are concatenated to a long vector
    t_end = L_bridge / v  # type: ignore
    t = np.arange(0, t_end, dt)
    if t[-1] != t_end:
        t = np.append(t, t[-1] + dt)
    nt = len(t)
    inp_1 = {"v": v, "t": t, "L": L_bridge, "F": F, "EI": EI_true}
    mean_dict = bridge_model.response(inp_1)
    mean = np.zeros(ns * nt)
    for ii, mean_vector in enumerate([*mean_dict.values()]):
        mean[ii::ns] = mean_vector

    # compute the covariance matrix using tripy
    cov_compiler = MeasurementSpaceTimePoints()
    cov_compiler.add_measurement_space_points(
        coord_mx=[[x] for x in x_sensors],
        standard_deviation=sigma,
        group="space",
    )
    cov_compiler.add_measurement_time_points(coord_vec=t, group="time")
    with HiddenPrints():  # this prevents printing of info messages from tripy
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

    # save the data for later
    data_dict[exp_name] = {"t": t, "v": v, "F": F}
    for ii in range(ns):
        data_dict[exp_name][f"y{ii + 1}"] = y_test[ii::ns]

    # first sensor
    c = exp_dict["plot_color"]
    plt.plot(t, mean[0::ns], "-", label=f"y1 (true, {exp_name})", color=c)
    plt.scatter(t, y1, marker="o", label=f"y1 (sampled, {exp_name})", c=c)

    # second sensor
    plt.plot(t, mean[1::ns], "--", label=f"y2 (true, {exp_name})", color=c)
    plt.scatter(t, y2, marker="x", label=f"y2 (sampled, {exp_name})", c=c)

# finish and show the plot
plt.title("Data of first two sensors for all experiments")
plt.xlabel("t [s]")
plt.ylabel("deflection [m]")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %%
# At this point we have some data to calibrate our model against. Hence, we can set up
# the inverse problem itself. This always begins by initializing an object form the
# InverseProblem-class and adding all of the problem's parameters with priors that
# reflect our current best guesses of what the parameter's values might look like.
# Please check out the 'Components'-part of this documentation to get more information
# on the arguments seen below. However, most of the code should be self-explanatory.

# initialize the inverse problem with a useful name
problem = InverseProblem(
    "Simple bridge model with time-space correlation", print_header=False
)

# add all parameters to the problem
problem.add_parameter(
    "EI",
    "model",
    domain="(0, +oo)",
    tex="$EI$",
    info="Bending stiffness of the beam [Nm^2]",
    prior=Normal(mean=0.9 * EI_true, std=0.25 * EI_true),
)
problem.add_parameter(
    "L", "model", tex="$L$", info="Length of the beam [m]", const=L_bridge
)
problem.add_parameter(
    "sigma",
    "likelihood",
    domain="(0, +oo)",
    tex=r"$\sigma$",
    info="Std. dev, of 0-mean noise model",
    prior=Uniform(low=low_sigma, high=high_sigma),
)
problem.add_parameter(
    "l_corr_x",
    "likelihood",
    domain="(0, +oo)",
    tex=r"$l_\mathrm{corr,x}$",
    info="Spatial correlation length of correlation model",
    prior=Uniform(low=low_l_corr_x, high=high_l_corr_x),
)
problem.add_parameter(
    "l_corr_t",
    "likelihood",
    domain="(0, +oo)",
    tex=r"$l_\mathrm{corr,t}$",
    info="Temporal correlation length of correlation model",
    prior=Uniform(low=low_l_corr_t, high=high_l_corr_t),
)

# %%
# As the next step, we need to add our forward model, the experimental data and the
# likelihood model. Note that the order is important and cannot be changed.

# experimental data
for exp_name, data in data_dict.items():

    sensor_values_vtF = {"v": data["v"], "t": data["t"], "F": data["F"]}
    sensor_values_x = {f"x{k + 1}": x_sensors[k] for k in range(ns)}
    sensor_values_y = {f"y{k + 1}": data[f"y{k + 1}"] for k in range(ns)}
    sensor_values = {**sensor_values_vtF, **sensor_values_x, **sensor_values_y}

    problem.add_experiment(
        name=exp_name,
        sensor_data=sensor_values,
    )

# add the forward model to the problem
problem.add_forward_model(bridge_model, experiments=[*data_dict.keys()])

# likelihood models
for exp_name in problem.experiments.keys():
    loglike = GaussianLikelihoodModel(
        experiment_name=exp_name,
        model_error="additive",
        correlation=ExpModel(x="l_corr_x", t="l_corr_t"),
    )
    problem.add_likelihood_model(loglike)

# %%
# Now, our problem definition is complete, and we can take a look at its summary:

# print problem summary
problem.info(print_header=True)

# %%
# After the problem definition comes the problem solution. There are different solver
# one can use, but we will just demonstrate how to use two of them: the scipy-solver,
# which merely provides a point estimate based on a maximum likelihood optimization, and
# the emcee solver, which is a MCMC-sampling solver. Let's begin with the scipy-solver:

# this is for using the scipy-solver (maximum likelihood estimation)
scipy_solver = ScipySolver(problem, show_progress=False)
max_like_data = scipy_solver.run_max_likelihood()

# %%
# All solver have in common that they are first initialized, and then execute a
# run-method, which returns its result data in the format of an arviz inference-data
# object (except for the scipy-solver). Let's now take a look at the emcee-solver.

emcee_solver = EmceeSolver(problem, show_progress=False)
inference_data = emcee_solver.run_emcee(n_steps=2000, n_initial_steps=200)

# %%
# Finally, we want to plot the results we obtained. To that end, probeye provides some
# post-processing routines, which are mostly based on the arviz-plotting routines.

# this is optional, since in most cases we don't know the ground truth
true_values = {
    "EI": EI_true,
    "sigma": sigma,
    "l_corr_x": l_corr_x,
    "l_corr_t": l_corr_t,
}

# this is an overview plot that allows to visualize correlations
pair_plot_array = create_pair_plot(
    inference_data,
    problem,
    true_values=true_values,
    focus_on_posterior=True,
    title="Sampling results from emcee-Solver (pair plot)",
)

# %%

# this is a posterior-focused plot, without including priors
post_plot_array = create_posterior_plot(
    inference_data,
    problem,
    true_values=true_values,
    title="Sampling results from emcee-Solver (posterior plot)",
)

# %%

# trace plots are used to check for "healthy" sampling
trace_plot_array = create_trace_plot(
    inference_data,
    problem,
    title="Sampling results from emcee-Solver (trace plot)",
)
