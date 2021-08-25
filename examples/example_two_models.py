"""
Inference problem with two forward models that share a common parameter
--------------------------------------------------------------------------------
The first model equation is y = a * x + b with a, b being the model parameters
and the second model equation is y = alpha * x**2 + b where alpha is a new model
parameter, and b is the same model parameter as in the first model equation.
Both forward models have the same noise model with a normal zero-mean
distribution where the standard deviation is to be inferred.The problem is
solved via sampling using taralli.
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

# 'true' value of b, and its normal prior parameters
b_true = 1.7
loc_b = 1.0
scale_b = 1.0

# 'true' value of alpha, and its normal prior parameters
alpha_true = 0.7
loc_alpha = 2.0
scale_alpha = 1.0

# 'true' value of sigma, and its normal prior parameters
sigma_true = 0.15
low_sigma = 0.1
high_sigma = 2.0

# the number of generated experiments and seed for random numbers
n_tests = 100
seed = 1

# taralli settings
n_walkers = 20
n_steps = 1000

# ============================================================================ #
#                          Define the Forward Models                           #
# ============================================================================ #

class LinearModel(ModelTemplate):
    def response(self, inp, sensor):
        x = inp['x']
        a = inp['a']
        b = inp['b']
        return a * x + b

class QuadraticModel(ModelTemplate):
    def response(self, inp, sensor):
        x = inp['x']
        alpha = inp['alpha']
        beta = inp['beta']
        return alpha * x**2 + beta

# ============================================================================ #
#                         Define the Inference Problem                         #
# ============================================================================ #

# initialize the inference problem with a useful name
problem = InferenceProblem("Two models with shared parameter and normal noise")

# add all parameters to the problem
problem.add_parameter('a', 'model',
                      info="Slope of the graph in linear model",
                      tex='$a$ (linear)',
                      prior=('normal', {'loc': loc_a,
                                        'scale': scale_a}))
problem.add_parameter('alpha', 'model',
                      info="Factor of quadratic term",
                      tex=r'$\alpha$ (quad.)',
                      prior=('normal', {'loc': loc_alpha,
                                        'scale': scale_alpha}))
problem.add_parameter('b', 'model',
                      info="Intersection of graph with y-axis",
                      tex='$b$ (shared)',
                      prior=('normal', {'loc': loc_b,
                                        'scale': scale_b}))
problem.add_parameter('sigma', 'noise',
                      tex=r"$\sigma$ (noise)",
                      info="Standard deviation of zero-mean noise model",
                      prior=('uniform', {'low': low_sigma,
                                         'high': high_sigma}))

# add the forward model to the problem
inp_1 = InputSensor("x")
out_1 = OutputSensor("y")
linear_model = LinearModel(['a', 'b'], [inp_1], [out_1])
problem.add_forward_model("LinearModel", linear_model)
quadratic_model = QuadraticModel(['alpha', {'b': 'beta'}], [inp_1], [out_1])
problem.add_forward_model("QuadraticModel", quadratic_model)

# add the noise model to the problem
problem.add_noise_model(out_1.name, NormalNoiseZeroMean(['sigma']))

# ============================================================================ #
#                    Add test data to the Inference Problem                    #
# ============================================================================ #

# data-generation process; normal noise with constant variance around each point
np.random.seed(seed)
x_test = np.linspace(0.0, 1.0, n_tests)
y_linear_true = linear_model({'x': x_test}, {'a': a_true, 'b': b_true})['y']
y_test_linear = np.random.normal(loc=y_linear_true, scale=sigma_true)
y_quadratic_true = quadratic_model(
    {'x': x_test}, {'alpha': alpha_true, 'beta': b_true})['y']
y_test_quadratic = np.random.normal(loc=y_quadratic_true, scale=sigma_true)

# add the experimental data
problem.add_experiment(f'TestSeries_linear',
                       sensors={'x': x_test, 'y': y_test_linear},
                       fwd_model_name="LinearModel")
problem.add_experiment(f'TestSeries_quadratic',
                       sensors={'x': x_test, 'y': y_test_quadratic},
                       fwd_model_name="QuadraticModel")

# give problem overview
problem.info()

# plot the true and noisy data
plt.scatter(x_test, y_test_linear, label='measured data (linear)', s=10,
            c="red", zorder=10)
plt.plot(x_test, y_linear_true, label='true (linear)', c="black")
plt.scatter(x_test, y_test_quadratic, label='measured data (quadratic)',
            s=10, c="orange", zorder=10)
plt.plot(x_test, y_quadratic_true, label='true (quadratic)', c="blue")
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
