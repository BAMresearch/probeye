"""
Linear regression with 1D-correlation
=====================================

The n data points (y1, y2, ..., yn) generated for this example are sampled from an
n-variate normal distribution with mean values given by yi = a * xi + b with a, b being
the model parameters and x1, x2, ..., xi, ..., xn being predefined spatial x-coordinates
ranging from 0 to 1. The data points (y1, y2, ..., yn) are not independent but
correlated in x. The corresponding covariance matrix is defined based on an exponential
correlation function parameterized by the constant standard deviation sigma of the
n-variate normal distribution and a correlation length l_corr. Hence, the full model has
four parameters a, b, sigma, l_corr, all of which are inferred in this example using
maximum likelihood estimation as well as sampling via emcee.
"""

# %%
# First, let's import the required functions and classes for this example.

# third party imports
import numpy as np
import matplotlib.pyplot as plt
from tripy.utils import correlation_function
from tripy.utils import correlation_matrix

# local imports (problem definition)
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# local imports (problem solving)
from probeye.inference.scipy.solver import ScipySolver
from probeye.inference.emcee.solver import EmceeSolver

# local imports (inference data post-processing)
from probeye.postprocessing.sampling import create_pair_plot
from probeye.postprocessing.sampling import create_posterior_plot
from probeye.postprocessing.sampling import create_trace_plot

# %%
# We start by generating a synthetic data set from a known linear model to which we will
# add correlated noise. Afterwards, we will pretend to have forgotten the parameters of
# this ground-truth model and will instead try to recover them just from the data. The
# slope (a) and intercept (b) of the ground truth model are set to be:

# ground truth
a_true = 2.5
b_true = 1.7

# %%
# Now, let's generate a few data points that we contaminate with a Gaussian error:

# settings for the data generation
n_experiments = 3
n_points = 50
seed = 1
std_noise = 0.5
l_corr = 0.05

# first create the true values without an error model; these 'true' values will be the
# mean values for sampling from a multivariate normal distribution that accounts for the
# intended correlation
np.random.seed(seed)
x_test = np.linspace(0.0, 1.0, n_points)
y_true = a_true * x_test + b_true

# assemble the spatial covariance matrix
x_test_as_column_matrix = x_test.reshape((n_points, -1))
f_corr = lambda a: correlation_function(d=a, correlation_length=l_corr)
cov = std_noise ** 2 * correlation_matrix(x_test_as_column_matrix, f_corr)

# now generate the noisy test data including correlations; we assume here that
# there are n_experiments test series
data_dict = {}
for i in range(n_experiments):
    y_test_i = np.random.multivariate_normal(mean=y_true, cov=cov)
    data_dict[f"Test_{i}"] = y_test_i
    plt.scatter(x_test, y_test_i, label=f"measured data (test {i+1})", s=10, zorder=10)
# finish the plot
plt.plot(x_test, y_true, label="true model", c="black")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# %%
# Until this point, we didn't use probeye at all, since we just generate some data. In
# a normal use case, we wouldn't have to generate our data of course. Instead, it would
# be provided to us, for example as the result of some test series. As the first step in
# any calibration problem, one needs to have a parameterized model (in probeye such a
# model is called 'forward model') of which one assumes that it is able to describe the
# data at hand. In this case, if we took a look at the blue data points in the plot
# above without knowing the orange line, we might expect a simple linear model. It is
# now our job to describe this model within the probeye-framework. This is done by
# defining our own specific model class:
class LinearModel(ForwardModelBase):
    def definition(self):
        self.parameters = ["a", "b"]
        self.input_sensors = Sensor("x")
        self.output_sensors = Sensor("y")

    def response(self, inp: dict) -> dict:
        x = inp["x"]
        m = inp["a"]
        b = inp["b"]
        return {"y": m * x + b}


# %%
# First, note that this model class is based on the probeye class 'ForwardModelBase'.
# While this is a requirement, the name of the class can be chosen freely. As you can
# see, this class has a 'definition' and a 'response' method. In the 'definition' method
# we define that our model has two parameters, 'a' and 'b', next to one input and one
# output sensors, called 'x' and 'y' respectively. Keeping this definition in mind,
# let's now take a look at the 'response' method. This method describes the actual
# forward model evaluation. The method takes one dictionary as an input and returns one
# dictionary as its output. The input dictionary 'inp' will have the keys 'a', 'b' and
# 'x' because of the definitions given in self.definition. Analogously, the returned
# dictionary must have the key 'y', because we defined an output sensor with the name
# 'y'. Note that the entire interface of the 'response' method is described by the
# 'definition' method. Parameters and input sensors will be contained in the 'inp'
# dictionary, while the output sensors must be contained in the returned dictionary.

# %%
# After we now have defined our forward model, we can set up the inference problem
# itself. This always begins by initializing an object form the InferenceProblem-class,
# and adding all of the problem's parameters with priors that reflect our current best
# guesses of what the parameter's values might look like. Please check out the
# 'Components'-part of this documentation to get more information on the arguments seen
# below. However, most of the code should be self-explanatory.

# initialize the inference problem with a useful name
problem = InferenceProblem("Linear regression with 1D correlation", print_header=False)

# add the problem's parameters
problem.add_parameter(
    "a",
    "model",
    tex="$a$",
    info="Slope of the graph",
    prior=("normal", {"loc": 2.0, "scale": 1.0}),
)
problem.add_parameter(
    "b",
    "model",
    info="Intersection of graph with y-axis",
    tex="$b$",
    prior=("normal", {"loc": 1.0, "scale": 1.0}),
)
problem.add_parameter(
    "std_noise",
    "likelihood",
    domain="(0, +oo)",
    tex=r"$\sigma$",
    info="Standard deviation, of zero-mean Gaussian noise model",
    prior=("uniform", {"low": 0.1, "high": 0.8}),
)
problem.add_parameter(
    "l_corr",
    "likelihood",
    domain="(0, +oo)",
    tex=r"$l_\mathrm{corr}$",
    info="Correlation length of correlation model",
    prior=("uniform", {"low": 0.001, "high": 0.2}),
)

# %%
# As the next step, we need to add our forward model, the experimental data and the
# likelihood model. Note that the order is important and cannot be changed.

# add the forward model to the problem
linear_model = LinearModel()
problem.add_forward_model("LinearModel", linear_model)

# experimental data
for exp_name, y_test_i in data_dict.items():
    problem.add_experiment(
        exp_name,
        fwd_model_name="LinearModel",
        sensor_values={"x": x_test, "y": y_test_i},
    )

# likelihood model
for exp_name in data_dict:
    likelihood_model = GaussianLikelihoodModel(
        prms_def=[{"std_noise": "std_model"}, "l_corr"],
        correlation_variables="x",
        correlation_model="exp",
        experiment_names=exp_name,
        additive_model_error=True,
        multiplicative_model_error=False,
        additive_measurement_error=False,
    )
    problem.add_likelihood_model(likelihood_model)

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
inference_data = emcee_solver.run_mcmc(n_steps=2000, n_initial_steps=200)

# %%
# Finally, we want to plot the results we obtained. To that end, probeye provides some
# post-processing routines, which are mostly based on the arviz-plotting routines.

# this is optional, since in most cases we don't know the ground truth
true_values = {"a": a_true, "b": b_true, "std_noise": std_noise, "l_corr": l_corr}

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
