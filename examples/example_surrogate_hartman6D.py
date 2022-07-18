"""
Example of a Bayesian parameter estimation problem using surrogate
modeling with probeye.

The Hartmann test function f:[0, 1]^6 -> R^1 is used to simulate a
physical model. The last two dimensions are considered as space and
time coordinates, while the first four dimensions are taken as
latent variables to be inferred. Measurements are generated by
adding I.i.d. Gaussian noise to samples from this function.

Notes:
    * torch, botorch and harlow are required to run this example
    * the sampling and surrogating package `harlow` will be made
     public soon
"""

# =========================================================================
# Imports
# =========================================================================

# third party imports
import numpy as np

# local imports (problem definition)
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.inference.emcee.solver import EmceeSolver
from probeye.definition.distribution import Uniform

# local imports (inference data post-processing)
from probeye.postprocessing.sampling_plots import create_pair_plot
from probeye.postprocessing.sampling_plots import create_posterior_plot

# Surrogate model imports
from harlow.sampling import Sampler
from harlow.sampling import LatinHypercube, FuzzyLolaVoronoi
from harlow.surrogating import Surrogate
from harlow.surrogating import ModelListGaussianProcess
from harlow.utils.transforms import ExpandDims

import torch
from botorch.test_functions.synthetic import Hartmann
from matplotlib import pyplot as plt

# =========================================================================
# General settings
# =========================================================================

plot = True

# Emcee settings
n_steps = 1_000
n_init_steps = 200
n_walkers = 20

# Surrogate settings
N_train = 50
N_train_iter = 200
show_progress = True

# =========================================================================
# Define parameters
# =========================================================================

# Ground truth
X_true = np.array([0.5, 0.5, 0.5, 0.5])

# Bounds for function defined on unit hypercube
X_low = 0.0
X_high = 1.0

# Ground truth and prior for measurement uncertainty std. dev.
std_true = 0.05
std_low = 0.0
std_high = 1.0

# =========================================================================
# Define physical model
# =========================================================================

# Number of sensors and number of points in timeseries
Nx = 1
Nt = 10

# Sensor names and positions
sensor_names = ["S" + str(i + 1) for i in range(Nx)]
x_vec = np.linspace(0, 1, Nx)
t_vec = np.linspace(0, 1, Nt)

isensor = Sensor("t")
osensor_list = [
    Sensor(sensor_names[i], x=float(x_vec[i]), std_model="sigma") for i in range(Nx)
]

# =========================================================================
# Define forward model
# =========================================================================

# Initialize model
expensive_model = Hartmann(noise_std=0.00001)


class SyntheticModel(ForwardModelBase):
    def interface(self):
        self.parameters = ["X" + str(i + 1) for i in range(4)]
        self.input_sensors = isensor
        self.output_sensors = osensor_list

    def response(self, inp: dict) -> dict:

        # Arange input vector
        params = np.tile([inp["X" + str(i + 1)] for i in range(4)], (Nx * Nt, 1))
        xt = np.array(np.meshgrid(x_vec, t_vec)).T.reshape(-1, 2)
        X = torch.tensor(np.hstack((params, xt)))

        # Evaluate function and arange output on grid
        f = np.array(expensive_model(X)).reshape(Nx, Nt)

        # Store prediction as dict
        response = dict()
        for idx_x, os in enumerate(self.output_sensors):
            response[os.name] = f[idx_x, :]
        return response


# =========================================================================
# Define surrogate model
# =========================================================================


def harlow_model_factory(
    problem: InverseProblem,
    forward_model: ForwardModelBase,
    surrogate_model: Surrogate,
    **kwargs,
) -> ForwardModelBase:
    class HarlowSurrogate(ForwardModelBase):
        """
        The inheritance from ExpensiveModel 'copies' the interface-method from
        ExpensiveModel (the surrogate model should have the same interface as the
        forward model).
        """

        def __init__(
            self,
            name,
        ):

            self.name = name
            self.kwargs = kwargs
            self.lower_bounds = []
            self.upper_bounds = []
            self.bounds = {}

            self.forward_model = forward_model
            self.problem = problem
            self._surrogate = surrogate_model

            self.model_names = []
            for idx_x, os in enumerate(forward_model.output_sensors):
                self.model_names.append(os.name)
            self.kwargs["model_names"] = self.model_names

            self.num_features = len(forward_model.output_sensors)
            self.kwargs["num_features"] = self.num_features

            # Check inputs
            if "input_transform" not in self.kwargs.keys():
                self.input_transform = None
            else:
                self.input_transform = self.kwargs["input_transform"]

            if "output_transform" not in self.kwargs.keys():
                self.output_transform = None
            else:
                self.output_transform = self.kwargs["output_transform"]

            # Initialize surrogate
            self.model = self._surrogate(**kwargs)
            self.initialize_forward_model()

        # Get interface from existing forward model
        def interface(self):
            self.input_sensors = self.forward_model.input_sensors
            self.output_sensors = self.forward_model.output_sensors
            self.parameters = self.forward_model.parameters

        def initialize_forward_model(self):
            super().__init__(self.name)
            self._get_bounds()

        # TODO: This will likely fail for multidimensional parameters. FIX!
        def _get_bounds(self):

            # Find the priors corresponding to model parameters
            for key, value in self.priors.items():

                # Error if any of the model priors are not uniform
                if value.prior_type != "uniform":
                    raise ValueError(
                        f"Non-uniform distribution of type {value.prior_type}"
                        f" found for prior {key}. The `HarlowSurrogate` currently"
                        f" only supports uniform priors."
                    )

            for param in self.problem.model_prms:
                lb = self.const_prms_dict["low_" + param]
                ub = self.const_prms_dict["high_" + param]
                self.bounds[param] = {"low": lb, "high": ub}

        def _cast_to_numpy(self, inp: dict) -> np.ndarray:
            raise NotImplementedError

        def _cast_to_dict(self, X: np.ndarray) -> dict:
            raise NotImplementedError

        def target(self, X: np.ndarray) -> np.ndarray:
            inp = self._cast_to_dict(X)
            response = self.forward_model.response(inp)
            y = [[]] * len(self.output_sensors)
            for idx, os in enumerate(self.output_sensors):
                y[idx] = response[os.name]
            return np.array(y).T

        def response(self, inp: dict) -> dict:

            params = np.tile([inp["X" + str(i + 1)] for i in range(4)], (Nt, 1))
            X = np.hstack((params, t_vec.reshape(-1, 1)))

            response = dict()

            # Evaluate function and arange output on grid
            f = self.model.predict(X, return_std=False)

            for idx_x, os in enumerate(self.output_sensors):
                response[os.name] = f[idx_x, :]

            return response

    return HarlowSurrogate


# =========================================================================
# Define inference problem
# =========================================================================
problem = InverseProblem("Parameter estimation using surrogate model")

# Parameters of the Hartmann function
for i in range(4):
    problem.add_parameter(
        "X" + str(i + 1),
        "model",
        prior=Uniform(low=X_low, high=X_high),
        info="Parameter of the 6D Hartmann function",
        tex=r"$X_{{{}}}$".format(i + 1),
    )

# Noise std. dev.
problem.add_parameter(
    "sigma",
    "likelihood",
    prior=Uniform(low=std_low, high=std_high),
    info="Std. dev. of zero-mean noise model",
    tex=r"$\sigma$",
)

# add the forward model to the problem
forward_model = SyntheticModel("ExpensiveModel")


# =========================================================================
# Create surrogate model
# =========================================================================

list_params = [[0, 1, 2, 3, 4]] * len(sensor_names)

surrogate_kwargs = {
    "training_max_iter": N_train_iter,
    "list_params": list_params,
    "model_names": [name for name in forward_model.sensor_names],
    "show_progress": True,
    "silence_warnings": True,
    "fast_pred_var": True,
}

# Generate surrogate class using model factory
surrogate_class = harlow_model_factory(
    problem, forward_model, ModelListGaussianProcess, **surrogate_kwargs
)

# Initialize surrogate model
surrogate_model = surrogate_class(
    name="FastModel",
)

# =========================================================================
# Add test data to the inference problem
# =========================================================================
def generate_data():
    inp = {"X" + str(idx + 1): X_i for idx, X_i in enumerate(X_true)}
    sensors = forward_model(inp)
    for sname, svals in sensors.items():
        sensors[sname] = list(
            np.array(svals) + np.random.normal(loc=0.0, scale=std_true, size=Nt)
        )
    sensors[isensor.name] = t_vec
    problem.add_experiment("TestSeriesFull", sensor_data=sensors)
    problem.add_experiment("TestSeriesSurrogate", sensor_data=sensors)


generate_data()

# Add Expensive FE model to forward models
problem.add_forward_model(forward_model, experiments="TestSeriesFull")
problem.add_forward_model(surrogate_model, experiments="TestSeriesSurrogate")

# =========================================================================
# Train surrogate model
# =========================================================================

# Initialize sampler
sampler = FuzzyLolaVoronoi(
    target_function=surrogate_model.target,
    surrogate_model=surrogate_model.model,
    domain_lower_bound=surrogate_model.lower_bound,
    domain_upper_bound=surrogate_model.upper_bound,
    test_points_x=test_X,
    test_points_y=test_y,
    evaluation_metric=rmse,
)
# main_start = time.time()
sampler.sample(
    n_iter=n_iter,
    n_initial_point=n_initial_point,
    n_new_point_per_iteration=n_new_points_per_iteration,
    ignore_old_neighborhoods=False,
    ignore_far_neighborhoods=False,
)

# # Train surrogate
# model_names = sensor_names
# surrogate_model.fit(train_X_arr, train_y_arr)


# ====================================================================
# Plot surrogate vs. FE model prediction
# ====================================================================

# Physical model prediction
inp = {"X" + str(idx + 1): X_i for idx, X_i in enumerate(X_true)}
y_true = forward_model.response(inp)
y_pred = surrogate_model.response(inp)
upper = list(surrogate_model.model.cr_u.squeeze().detach().numpy().T)
lower = list(surrogate_model.model.cr_l.squeeze().detach().numpy().T)

# Initialize plots
nrows = 3
ncols = int(np.ceil(len(sensor_names) / 3))

# Initialize plot
f, axes = plt.subplots(nrows, ncols, sharex=True, figsize=(3 * ncols, 3 * nrows))

for j in range(len(sensor_names)):

    ax_i = axes.ravel()[j]
    grid_idx = np.unravel_index(j, (nrows, ncols))

    # Plot training data as black stars
    ax_i.plot(t_vec, train_y[:, j, :].T, "k*", label="Observations")

    # Predictive mean as blue line
    ax_i.plot(t_vec, y_pred[sensor_names[j]], "b", label="Mean")

    # Shade in confidence
    ax_i.fill_between(
        t_vec,
        lower[j],
        upper[j],
        alpha=0.5,
        label="Confidence",
    )
    ax_i.plot(
        t_vec,
        y_true[sensor_names[j]],
        color="red",
        linestyle="dashed",
        label="Model",
    )
    ax_i.set_title(f"Sensor: {sensor_names[j]}")

axes = np.atleast_2d(axes)
# axes[0, 0].legend()
[ax_i.set_xlabel(r"$X_6$") for ax_i in axes[-1, :]]
[ax_i.set_ylabel("Stress [MPa]") for ax_i in axes[:, 0]]
plt.show()


# =========================================================================
# Add noise models
# =========================================================================

# add the likelihood models to the problem
for osensor in osensor_list:
    problem.add_likelihood_model(
        GaussianLikelihoodModel(
            experiment_name="TestSeriesSurrogate",
            prms_def=["sigma"],
            model_error="additive",
            additive_measurement_error=False,
            correlation_variables=None,
        )
    )


# Problem overview
problem.info()

true_values = {
    "X1": X_true[0],
    "X2": X_true[1],
    "X3": X_true[2],
    "X4": X_true[3],
    "sigma": std_true,
}

# =========================================================================
# Initialize and run solver
# =========================================================================

emcee_solver = EmceeSolver(
    problem,
    show_progress=True,
)


inference_data = emcee_solver.run_mcmc(
    n_walkers=n_walkers, n_steps=n_steps, n_initial_steps=n_init_steps, vectorize=False
)

# =========================================================================
# Plotting
# =========================================================================
create_pair_plot(
    inference_data,
    problem,
    show=False,
    true_values=true_values,
    title="Joint posterior",
)

create_posterior_plot(
    inference_data,
    problem,
    show=False,
    true_values=true_values,
    title="Marginal posteriors",
)


if plot:
    plt.show()  # shows all plots at once due to 'show=False' above
else:
    plt.close("all")
