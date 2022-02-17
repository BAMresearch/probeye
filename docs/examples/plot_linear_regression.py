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
import numpy as np
import matplotlib.pyplot as plt

# local imports (problem definition)
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# local imports (problem solving)
from probeye.inference.scipy.solver import ScipySolver
from probeye.inference.emcee.solver import EmceeSolver
from probeye.inference.torch.solver import PyroSolver
from probeye.inference.dynesty.solver import DynestySolver

# local imports (inference data post-processing)
from probeye.postprocessing.sampling import create_pair_plot
from probeye.postprocessing.sampling import create_posterior_plot
from probeye.postprocessing.sampling import create_trace_plot

# %%
# We start by generating a synthetic data set from a known linear model to which we will
# add some noise. Afterwards, we will pretend to have forgotten the parameters of this
# ground-truth model and will instead try to recover them just from the data. The slope
# (a) and intercept (b) of the ground truth model are set to be:

# ground truth
a_true = 2.5
b_true = 1.7

# %%
# Now, let's generate a few data points that we contaminate with a Gaussian error:

# settings for data generation
n_tests = 50
seed = 1
mean_noise = 0.0
std_noise = 0.5

# generate the data
np.random.seed(seed)
x_test = np.linspace(0.0, 1.0, n_tests)
y_true = a_true * x_test + b_true
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

# initialize the problem
problem = InferenceProblem("Linear regression with Gaussian noise", print_header=False)

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
    tex=r"$\sigma$",
    info="Standard deviation, of zero-mean Gaussian noise model",
    prior=("uniform", {"low": 0.1, "high": 0.8}),
)

# %%
# As the next step, we need to add our forward model, the experimental data and the
# likelihood model. Note that the order is important and cannot be changed.

# forward model
linear_model = LinearModel()
problem.add_forward_model("LinearModel", linear_model)

# experimental data
problem.add_experiment(
    "TestSeries_1",
    fwd_model_name="LinearModel",
    sensor_values={"x": x_test, "y": y_test},
)

# likelihood model
problem.add_likelihood_model(
    GaussianLikelihoodModel(
        prms_def={"std_noise": "std_model"}, sensors=linear_model.output_sensor
    )
)

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
# object (except for the scipy-solver). Let's now take a look at the emcee-solver. If
# you want to try out another solver, you just have to change the flags that follow
# below.

# solver flags
run_emcee = True
run_dynesty = False
run_pyro = False

# this is for running the dynesty-solver
if run_dynesty:
    dynesty_solver = DynestySolver(problem, show_progress=False)
    inference_data = dynesty_solver.run_dynesty()

# this is for using the pyro-solver (NUTS-sampler)
if run_pyro:
    pyro_solver = PyroSolver(problem, show_progress=False)
    inference_data = pyro_solver.run_mcmc(n_steps=2000, n_initial_steps=200)

# this is for using the emcee-solver
if run_emcee:
    emcee_solver = EmceeSolver(problem, show_progress=False)
    inference_data = emcee_solver.run_mcmc(n_steps=2000, n_initial_steps=200)

# %%
# Finally, we want to plot the results we obtained. To that end, probeye provides some
# post-processing routines, which are mostly based on the arviz-plotting routines.

# this is optional, since in most cases we don't know the ground truth
true_values = {"a": a_true, "b": b_true, "std_noise": std_noise}

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
