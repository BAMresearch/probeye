"""
Simple linear regression example with two model and one noise parameter
--------------------------------------------------------------------------------
The model equation is y = a * x + b with a, b being the model parameters and the
noise model is a normal zero-mean distribution with the std. deviation to infer.
The problem is solved via sampling using taralli.
"""
# ============================================================================ #
#                                   Imports                                    #
# ============================================================================ #

# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports
from probeye.forward_model import ModelTemplate
from probeye.forward_model import Sensor
from probeye.noise import NormalNoiseZeroMean
from probeye.inference_problem import InferenceProblem
from probeye.solver.taralli import taralli_solver

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

# 'true' value of noise sd, and its uniform prior parameters
sigma_noise = 0.5
low_sigma = 0.1
high_sigma = 0.8

# the number of generated experiments and seed for random numbers
n_tests = 50
seed = 1

# taralli settings
n_walkers = 20
n_steps = 1000

# ============================================================================ #
#                           Define the Forward Model                           #
# ============================================================================ #

class LinearModel(ModelTemplate):
    def __call__(self, inp):
        x = inp['x']
        m = inp['m']
        b = inp['b']
        return {'y': m * x + b}

# ============================================================================ #
#                         Define the Inference Problem                         #
# ============================================================================ #

# initialize the inference problem with a useful name; note that the name will
# only be stored as an attribute of the InferenceProblem and is not important
# for the problem itself; can be useful when dealing with multiple problems
problem = InferenceProblem("Linear regression with normal noise")

# add all parameters to the problem; the first argument states the parameter's
# global name (here: 'a', 'b' and 'sigma'); the second argument defines the
# parameter type (three options: 'model' for parameter's of the forward model,
# 'prior' for prior parameters and 'noise' for parameters of the noise model);
# the 'info'-argument is a short description string used for logging, and the
# tex-argument gives a tex-string of the parameter used for plotting; finally,
# the prior-argument specifies the parameter's prior; note that this definition
# of a prior will result in the initialization of constant parameters of type
# 'prior' in the background
problem.add_parameter('a', 'model',
                      info="Slope of the graph", tex="$a$",
                      prior=('normal', {'loc': loc_a, 'scale': scale_a}))
problem.add_parameter('b', 'model',
                      info="Intersection of graph with y-axis", tex='$b$',
                      prior=('normal', {'loc': loc_b, 'scale': scale_b}))
problem.add_parameter('sigma', 'noise',
                      info="Std. dev, of 0-mean noise model", tex=r"$\sigma$",
                      prior=('uniform', {'low': low_sigma, 'high': high_sigma}))

# add the forward model to the problem; note that the argument [{'a': 'm'}, 'b']
# passed to LinearModel defines the forward model's parameters by name via a
# list with elements like {<global parameter name>: <local parameter name>};
# a global name is a name introduced by problem.add_parameter, while a local
# name is a name used in the response-method of the forward model class (see
# the definition of the class LinearModel above); note that the use of the
# local parameter name 'm' for the global parameter 'a' is added here only to
# highlight the possibility of this feature; it is not necessary at all here;
# whenever forward model's parameter has a similar local and global name (which
# should be the case most of the times), one doesn't have to use the verbose
# notation {<global parameter name>: <local parameter name>} but can instead
# just write the parameter's (global=local) name, like it is done with the
# forward model's parameter 'b' below
inp_1 = Sensor("x")
out_1 = Sensor("y")
linear_model = LinearModel([{'a': 'm'}, 'b'], [inp_1], [out_1])
problem.add_forward_model("LinearModel", linear_model)

# add the noise model to the problem
problem.add_noise_model(out_1.name, NormalNoiseZeroMean(['sigma']))

# ============================================================================ #
#                    Add test data to the Inference Problem                    #
# ============================================================================ #

# data-generation process; normal noise with constant variance around each point
np.random.seed(seed)
x_test = np.linspace(0.0, 1.0, n_tests)
y_true = linear_model({'x': x_test, 'm': a_true, 'b': b_true})['y']
y_test = np.random.normal(loc=y_true, scale=sigma_noise)

# add the experimental data
problem.add_experiment(f'TestSeries_1', fwd_model_name="LinearModel",
                       sensor_values={'x': x_test, 'y': y_test})

# give problem overview
problem.info()

# plot the true and noisy data
plt.scatter(x_test, y_test, label='measured data', s=10, c="red", zorder=10)
plt.plot(x_test, y_true, label='true', c="black")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.draw()  # does not stop execution

# ============================================================================ #
#                          Solve problem with Taralli                          #
# ============================================================================ #

# code is bundled in a specific solver routine
taralli_solver(problem, n_walkers=n_walkers, n_steps=n_steps)
