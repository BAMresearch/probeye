"""
Linear regression example where a prior parameter is a calibration parameter
--------------------------------------------------------------------------------
The model equation is y = a * x + b with a, b being the model parameters and the
noise model is a normal zero-mean distribution with the std. deviation to infer.
Additionally, the location parameter of a's prior is considered a calibration
parameter.The problem is solved via sampling using taralli.
"""
# ============================================================================ #
#                                   Imports                                    #
# ============================================================================ #

# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports
from probeye.forward_model import ModelTemplate
from probeye.forward_model import InputSensor, OutputSensor
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

# uniform prior-parameters of 'loc_a'
low_loc_a = 2.0
high_loc_a = 3.0

# 'true' value of b, and its normal prior parameters
b_true = 1.7
loc_b = 1.0
scale_b = 1.0

# 'true' value of noise sd, and its uniform prior parameters
sigma_noise = 0.5
low_sigma = 0.1
high_sigma = 0.6

# the number of generated experiments and seed for random numbers
n_tests = 100
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
        a = inp['a']
        b = inp['b']
        return {'y': a * x + b}

# ============================================================================ #
#                         Define the Inference Problem                         #
# ============================================================================ #

# initialize the inference problem with a useful name
problem = InferenceProblem("Linear model with normal noise and prior-prior")

# add all parameters to the problem
problem.add_parameter('a', 'model',
                      info="Slope of the graph", tex="$a$",
                      prior=('normal', {'loc': loc_a, 'scale': scale_a}))
problem.add_parameter('b', 'model',
                      info="Intersection of graph with y-axis", tex='$b$',
                      prior=('normal', {'loc': loc_b, 'scale': scale_b}))
problem.add_parameter('sigma', 'noise',
                      info="Std. dev, of 0-mean noise model", tex=r"$\sigma$",
                      prior=('uniform', {'low': low_sigma, 'high': high_sigma}))

# this makes the prior-parameter 'loc_a' a calibration parameter
problem.change_parameter_role('loc_a', new_tex=r"$\mu_a^\mathrm{prior}$",
                              new_info="Std. deviation of normal prior for 'a'",
                              prior=('uniform', {'low': low_loc_a,
                                                 'high': high_loc_a}))

# add the forward model to the problem
inp_1 = InputSensor("x")
out_1 = OutputSensor("y")
linear_model = LinearModel(['a', 'b'], [inp_1], [out_1])
problem.add_forward_model("LinearModel", linear_model)

# add the noise model to the problem
problem.add_noise_model(out_1.name, NormalNoiseZeroMean(['sigma']))

# ============================================================================ #
#                    Add test data to the Inference Problem                    #
# ============================================================================ #

# data-generation process; normal noise with constant variance around each point
np.random.seed(seed)
x_test = np.linspace(0.0, 1.0, n_tests)
y_true = linear_model({'x': x_test, 'a': a_true, 'b': b_true})['y']
y_test = np.random.normal(loc=y_true, scale=sigma_noise)

# add the experimental data
problem.add_experiment(f'TestSeries_1', sensors={'x': x_test, 'y': y_test},
                       fwd_model_name="LinearModel")

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
