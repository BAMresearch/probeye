# standard library imports
import unittest

# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports (problem definition)
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.noise_model import NormalNoiseModel

# local imports (testing related)
from tests.integration_tests.subroutines import run_inference_engines

# ==================================================================== #
#                          Set numeric values                          #
# ==================================================================== #

# the acceleration due to gravity [m/s^2]
g_true = 9.84
loc_g = 9.8
scale_g = 0.1

# the number of generated experiment_names and seed for random numbers
n_tests = 50
seed = 1

# ==================================================================== #
#                       Data generation process                        #
# ==================================================================== #

v_0 = 20.0

def forward_process(t, v_0, g=9.81, m=0.145, r=0.036, rho=1.29, cd=0.05):
    """
    Computes the trajectory of a ball that is thrown straight upwards from
    ground level (zero height), until it falls back on the ground. The model
    accounts for Newtonian air drag and - in this example - should represent
    the real (instead of the modeled) process. The formulas have been taken
    from https://en.wikipedia.org/wiki/Projectile_motion.

    Parameters
    ----------
    t : float
        The time for which the position (i.e, the height) should be evaluated.
    v_0 : float
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
    y : float
        The elevation of the ball in the specified scenario at given time t.
    """

    # derived parameters
    mu = 0.5 * cd * (np.pi * r ** 2) * rho / m
    t_f = 1 / np.sqrt(g * mu)
    v_inf = np.sqrt(g / mu)

    # compute time and elevation when the ball reaches its highest point
    t_peak = t_f * np.arctan(v_0 / v_inf)
    y_peak = -1 / mu * np.log(np.cos(t_peak / t_f))

    # compute the time when the ball falls back on the ground
    # t_end = t_peak + t_f * np.arccosh(np.exp(mu * y_peak))

    # there are two different equations for the upward and downward direction,
    # so we have to do a check here, to see on which side we are
    if t <= t_peak:
        y = y_peak + 1 / mu * np.log(np.cos((t_peak - t) / t_f))
    else:
        y = y_peak - 1 / mu * np.log(np.cosh((t - t_peak) / t_f))
    return y

t_up = np.linspace(0, t_peak, 100)
t_down = np.linspace(t_peak, t_end, 100)
y_up = y_peak + 1 / mu * np.log(np.cos((t_peak - t_up) / t_f))
y_down = y_peak - 1 / mu * np.log(np.cosh((t_down - t_peak) / t_f))

t = np.linspace(0, t_end)
y_model = v_0 * t - 0.5 * g * t ** 2

# plot the true and noisy data
plt.plot(t_up, y_up, label='up')
plt.plot(t_down, y_down, label='down')
plt.plot(t, y_model, label='model')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()

