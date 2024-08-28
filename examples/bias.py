"""
Simple linear regression example
================================

The model equation is y = ax + b with a, b being the model parameters, while the
likelihood model is based on a normal zero-mean additive model error distribution with
the standard deviation to infer. The problem is solved via maximum likelihood estimation
as well as via sampling using emcee.
"""

# %%
# First, let's import the required functions and classes for this example.

# third party imports
import math
import numpy as np
import matplotlib.pyplot as plt
import chaospy

# local imports (problem definition)
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.distribution import Normal, Uniform, LogNormal
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.inference.bias.likelihood_models import (
    MomentMatchingModelError,
    GlobalMomentMatchingModelError,
    RelativeGlobalMomentMatchingModelError,
    IndependentNormalModelError,
)

# local imports (problem solving)
from probeye.inference.scipy.solver import MaxLikelihoodSolver
from probeye.inference.emcee.solver import EmceeSolver
from probeye.inference.dynesty.solver import DynestySolver
from probeye.inference.bias.solver import EmbeddedPCESolver

# local imports (inference data post-processing)
from probeye.postprocessing.sampling_plots import create_pair_plot
from probeye.postprocessing.sampling_plots import create_posterior_plot
from probeye.postprocessing.sampling_plots import create_trace_plot

# %%
# We start by generating a synthetic data set from a known linear model to which we will
# add some noise in the slope parameter. Afterwards, we will pretend to have forgotten
# the parameters of this ground-truth model and will instead try to recover them just
# from the data. The slope (a), the parameter bias scale (b) and the noise scale of the
# ground truth model are set to be:

# ground truth
a_true = 4.0
b_true = 1.0
noise_std = 0.01

# %%
# Now, let's generate a few data points that we contaminate with a prescribed noise:

# settings for data generation
n_tests = 50
seed = 1
mean_noise = 0.0
std_noise = np.linspace(0.2 * b_true, b_true, n_tests)
std_noise += np.random.normal(loc=0.0, scale=noise_std, size=n_tests)

# generate the data
np.random.seed(seed)
x_test = np.linspace(0.2, 1.0, n_tests)
y_true = a_true * x_test
y_test = y_true + np.random.normal(loc=mean_noise, scale=std_noise, size=n_tests)

# %%
# Let's take a look at the data that we just generated:
plt.plot(x_test, y_test, "o", label="generated data points")
plt.plot(x_test, y_true, label="ground-truth model")
plt.title("Data vs. ground truth")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# %%
# We define a linear model that uses polynomial chaos expansions (PCEs) to approximate
# the model response. The model is defined as a class that inherits from the
# `ForwardModelBase` class. The `interface` method is used to define the model's
# parameters, input sensors, and output sensors. The `response` method is used to
# calculate the model's response based on the input parameters. In this case, the model
# is a simple linear regression model with slope `a` and bias scale `b`. The bias term
# is modeled as a normal distribution with zero mean and standard deviation `b`. The
# model response is calculated using PCEs with pseudo-spectral decomposition.


class LinearModel(ForwardModelBase):
    def interface(self):
        self.parameters = ["a", "b"]
        self.input_sensors = Sensor("x")
        self.output_sensors = Sensor("y", std_model="sigma")

    def response(self, inp: dict) -> dict:
        pce_order = 1
        x = inp["x"]
        m = inp["a"]
        b = inp["b"]

        m = np.repeat(m, len(x))
        x = x.reshape(-1, 1)
        m = m.reshape(-1, 1)

        # define the distribution for the bias term
        b_dist = chaospy.Normal(0.0, b)
        # generate quadrature nodes and weights
        sparse_quads = chaospy.generate_quadrature(pce_order, b_dist, rule="Gaussian")
        # evaluate the model at the quadrature nodes
        sparse_evals = np.array(
            [np.array((m + node) * x) for node in sparse_quads[0][0]]
        )
        # generate the polynomial chaos expansion
        expansion = chaospy.generate_expansion(pce_order, b_dist)
        # fit the polynomial chaos expansion
        fitted_sparse = chaospy.fit_quadrature(
            expansion, sparse_quads[0], sparse_quads[1], sparse_evals
        )
        return {"y": fitted_sparse, "dist": b_dist}


# %%
# We initialize the inverse problem by providing a name and some additional information
# in the same way as for the other examples. The bias parameter is defined as any other
# parameter. In this case, noise is prescribed.
problem = InverseProblem("Linear regression with embedding", print_header=False)

# add the problem's parameters
problem.add_parameter(
    "a",
    tex="$a$",
    info="Slope of the graph",
    prior=Normal(mean=3.5, std=0.5),
    domain="(3, 6.0)",
)

problem.add_parameter(
    "b",
    tex="$b$",
    info="Standard deviation of the bias",
    prior=LogNormal(mean=-1.0, std=0.5),
    # domain="(-1, 4)",
)

problem.add_parameter(
    "sigma",
    tex=r"$\sigma$",
    info="Standard deviation, of zero-mean Gaussian noise model",
    value=noise_std,
)

# %%
# As the next step, we need to add our experimental data the forward model and the
# likelihood model as for the other examples. Note that some of the embedded models
# require the specification of tolerance and gamma parameters.

# experimental data
problem.add_experiment(
    name="TestSeries_1",
    sensor_data={"x": x_test, "y": y_test},
)

problem.add_forward_model(LinearModel("LinearModel"), experiments="TestSeries_1")

dummy_lmodel = GaussianLikelihoodModel(
    experiment_name="TestSeries_1", model_error="additive"
)
likelihood_model = IndependentNormalModelError(dummy_lmodel)
problem.add_likelihood_model(likelihood_model)

# %%
# Now, our problem definition is complete, and we can take a look at its summary:

# print problem summary
problem.info(print_header=True)

# %%
# To solve the problem, we use the specialized solver for embedded models using PCEs.

solver = EmbeddedPCESolver(problem, show_progress=False)
inference_data = solver.run(n_steps=200, n_initial_steps=20)

# %%
# Finally, we want to plot the results we obtained.
true_values = {"a": a_true, "b": b_true}

# this is an overview plot that allows to visualize correlations
pair_plot_array = create_pair_plot(
    inference_data,
    solver.problem,
    true_values=true_values,
    focus_on_posterior=True,
    show_legends=True,
    title="Sampling results from emcee-Solver (pair plot)",
)

# %%

# this is a posterior plot, without including priors
post_plot_array = create_posterior_plot(
    inference_data,
    solver.problem,
    true_values=true_values,
    title="Sampling results from emcee-Solver (posterior plot)",
)

# %%

# trace plots are used to check for "healthy" sampling
trace_plot_array = create_trace_plot(
    inference_data,
    solver.problem,
    title="Sampling results from emcee-Solver (trace plot)",
)

# %%
# Plot posterior results with estimated parameters. The PCE must be rebuilt with the
# estimated parameters to obtain the fitted model. This is done in the forward model's
# response method.

mean_a = np.mean(inference_data.posterior["$a$"].values)
mean_b = np.mean(inference_data.posterior["$b$"]).values

fitted_model_input = {"x": x_test, "a": mean_a, "b": mean_b}
forward_model = LinearModel("LinearModel")
fitted_model_output = forward_model.response(fitted_model_input)
output_mean = chaospy.E(fitted_model_output["y"], fitted_model_output["dist"])
output_std = chaospy.Std(fitted_model_output["y"], fitted_model_output["dist"])

figure_1 = plt.figure()
ax_1 = figure_1.add_subplot(111)
plt.plot(x_test, y_test, "ko", label="Generated data points")
plt.plot(x_test, output_mean, "g", label="Fitted model")
plt.plot(x_test, output_mean - noise_std, "r--", label=r"Fitted model $\pm \sigma_N$")
plt.plot(x_test, output_mean + noise_std, "r--")
plt.plot(
    x_test,
    output_mean - np.sqrt(output_std**2 + noise_std**2),
    "b--",
    label=r"Fitted model $\pm \sqrt{\sigma^2+\sigma_N^2}$",
)
plt.plot(x_test, output_mean + np.sqrt(output_std**2 + noise_std**2), "b--")
plt.title("Fitted model predictions")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# %%
