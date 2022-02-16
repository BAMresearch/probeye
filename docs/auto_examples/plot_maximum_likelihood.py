"""
Simple linear regression (maximum likelihood)
=============================================

A simple linear regression example with two model parameters and one noise parameter.

The model equation is y = a * x + b with a, b being the model parameters and the
noise model is a normal zero-mean distribution with the std. deviation to infer.
The problem is solved via maximum likelihood estimation.
"""

# %%
# Import what we will need for this example.

# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports (problem definition)
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# local imports (inference related)
from probeye.inference.scipy.solver import ScipySolver

# %%
# We start by generating a synthetic data set from a known linear model. Later we
# will pretend to forgot about the parameters of this ground truth model and will try
# to recover them from the data. The slope and intercept of the ground truth model:

a_true = 2.5
b_true = 1.7

# %%
# Generate a few data points that we contaminate with a Gaussian error:
n_tests = 50
seed = 1

np.random.seed(seed)
x_test = np.linspace(0.0, 1.0, n_tests)
y_true = a_true * x_test + b_true
sigma_noise = 0.5
y_test = y_true + np.random.normal(loc=0, scale=sigma_noise, size=n_tests)

# %%
# Visualize our data points
plt.plot(x_test, y_test, "o")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# %%
# Define our parametrized linear model:
class LinearModel(ForwardModelBase):
    def response(self, inp):
        # this method *must* be provided by the user
        x = inp["x"]
        m = inp["m"]
        b = inp["b"]
        response = {}
        for os in self.output_sensors:
            response[os.name] = m * x + b
        return response


# %%
# Define the inference problem.
# Initialize the inference problem with a useful name; note that the
# name will only be stored as an attribute of the InferenceProblem and
# is not important for the problem itself; can be useful when dealing
# with multiple problems

problem = InferenceProblem("Linear regression with normal noise")

# %%
# Add all parameters to the problem; the first argument states the
# parameter's global name (here: 'a', 'b' and 'sigma'); the second
# argument defines the parameter type (three options: 'model' for
# parameter's of the forward model, 'prior' for prior parameters and
# 'noise' for parameters of the noise model); the 'info'-argument is a
# short description string used for logging, and the tex-argument gives
# a tex-string of the parameter used for plotting; finally, the prior-
# argument specifies the parameter's prior; note that this definition
# of a prior will result in the initialization of constant parameters of
# type 'prior' in the background
problem.add_parameter("a", "model", tex="$a$")
problem.add_parameter("b", "model", tex="$b$")
problem.add_parameter("sigma", "likelihood", tex=r"$\sigma$")

# %%
# Add the forward model to the problem; note that the first positional
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
linear_model = LinearModel([{"a": "m"}, "b"], [isensor], [osensor])
problem.add_forward_model("LinearModel", linear_model)

# %%
# Add test data to the Inference Problem
problem.add_experiment(
    "TestSeries_1",
    fwd_model_name="LinearModel",
    sensor_values={isensor.name: x_test, osensor.name: y_test},
)

# %%
# add the likelihood model to the problem
problem.add_likelihood_model(
    GaussianLikelihoodModel(prms_def={"sigma": "std_model"}, sensors=osensor)
)

# give problem overview
problem.info()

# %%
# Estimate the parameters using `Scipy`
scipy_solver = ScipySolver(problem, show_progress=True)
inference_data = scipy_solver.run_max_likelihood()
