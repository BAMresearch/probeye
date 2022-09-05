import numpy as np
from typing import Callable

# local imports
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase

# Harlow imports
from harlow.sampling import Sampler
from harlow.surrogating import Surrogate
from harlow.utils.helper_functions import latin_hypercube_sampling


class SurrogateModelBase(ForwardModelBase):
    """
    Base class for a surrogate model, i.e., a forward model that approximates another
    (typically computationally more expensive) forward model.

    Parameters
    ----------
    name
        The name of the surrogate model. Must be unique among all surrogate model's
        names within a considered InverseProblem.
    forward_model
        The forward model object that the surrogate model approximates.
    """

    def __init__(self, name: str, forward_model: ForwardModelBase):

        super().__init__(name)

        # the surrogate model has access to the forward model it approximates; this
        # forward model can be called (evaluating its response) from the surrogate model
        # via self.forward_model.response(inp) where inp contains the input dictionary
        self.forward_model = forward_model

    def fit(self):
        """
        Prepares the surrogate model by approximating the forward model in some way.
        """
        pass


class HarlowModelFactory:
    """
    Model factory that returns a `ForwardModel` object, where the response
    is obtained from a fitted surrogate model.

    Usage:
    TODO

    Todo:
    * Currently, each point is added as a separate surrogate model

    Notes:
    * This is a utility class meant to facilitate the sampling and surrogating
    of expensive forward models by bridging `probeye` with `harlow`. The
    main objective is to reduce the manual effort required for surrogating
    by eliminating the need to re-specify the problem when building the
    surrogate.

    """

    def __init__(
        self,
        problem: InverseProblem,
        forward_model: ForwardModelBase,
        sampler: Sampler,
        surrogate_model: Surrogate,
        fit_points_x: np.ndarray = None,
        fit_points_y: np.ndarray = None,
        test_points_x: np.ndarray = None,
        test_points_y: np.ndarray = None,
        evaluation_metric: Callable = None,
        verbose: bool = False,
        run_name: str = "test",
        **surrogate_kwargs,
    ):

        # Initialize
        self.surrogate_kwargs = surrogate_kwargs
        self.problem = problem
        self.forward_model = forward_model
        self.input_sensors = forward_model.input_sensors
        self.output_sensors = forward_model.output_sensors
        self.parameters = forward_model.parameters
        self.fit_points_x = fit_points_x
        self.fit_points_y = fit_points_y
        self.test_points_x = test_points_x
        self.test_points_y = test_points_y
        self.evaluation_metric = evaluation_metric
        self.verbose = verbose
        self.run_name = run_name

        # Find the experiment associated with the expensive forward model
        # and get sensor information
        # TODO: Implement check that the expensive forward model
        #  has been added to the problem
        self.input_sensor_data = dict()
        for exp_name, exp in self.problem.experiments.items():
            if exp.forward_model == self.forward_model.name:
                for inp_sensor in self.input_sensors:
                    self.input_sensor_data[inp_sensor.name] = list(
                        exp.sensor_data[inp_sensor.name]
                    )

        # Check that the supplied forward model only has a single sensor
        # TODO: Can a forward model have multiple input sensors?
        if len(list(self.input_sensor_data.keys())) != 1:
            raise ValueError(
                f"The specified forward model must have "
                f"exactly one input sensor but has "
                f"{len(list(self.input_sensor_data.keys()))}"
            )

        # Extract sensor names and input/output shape information
        # from forward model.
        # TODO: This should be adjusted depending on the problem definition
        #   or specified by the user e.g. with keyword arguments that specify
        #   the structure of the surrogate model, e.g.:
        #   * `shape = None` to create an independent surrogate for each observation
        #   * `shape = Sensor` to create a surrogate model per sensor
        self.vec_t = list(self.input_sensor_data.values())[0]
        self.model_names = []
        for idx_x, os in enumerate(forward_model.output_sensors):
            for idx_t, t in enumerate(self.vec_t):
                self.model_names.append(os.name + "_" + str(t))
        self.surrogate_kwargs["model_names"] = self.model_names
        self.num_features = len(forward_model.output_sensors)
        self.surrogate_kwargs["num_features"] = self.num_features

        # Get bounds
        self._get_bounds()

        # Initialize surrogate
        self.surrogate = surrogate_model(**surrogate_kwargs)
        self.func_pred = self.surrogate.predict

        # Initialize sampler
        self.sampler = sampler(
            target_function=self.target,
            surrogate_model=self.surrogate,
            domain_lower_bound=self.lower_bounds,
            domain_upper_bound=self.upper_bounds,
            fit_points_x=self.fit_points_x,
            fit_points_y=self.fit_points_y,
            test_points_x=self.test_points_x,
            test_points_y=self.test_points_y,
            evaluation_metric=self.evaluation_metric,
            verbose=self.verbose,
            run_name=self.run_name,
        )

    def fit(self, **kwargs):
        """
        Fit the surrogate model using the sampling points and observations
        stored in `self.sampler`

        Parameters
        ----------
        kwargs

        Returns
        -------

        """

        self.surrogate.fit(
            self.sampler.fit_points_x, self.sampler.fit_points_y, **kwargs
        )

    def sample(self, **kwargs):
        """
        Draw samples from the forward model and fit the surrogate model
        using the specified sampler.

        Parameters
        ----------
        kwargs

        Returns
        -------

        """

        self.fit_points_x, self.fit_points_y = self.sampler.sample(**kwargs)

        # # Get prediction function
        # self.func_pred = self.surrogate.predict

    def get_training_data(self, n_samples=1):
        """
        Obtain training points `X` with latin hypercube sampling and
        corresponding observations `y` by evaluating the target function.

        Returns
        -------
        A tuple of training points and observations `(train_X, train_y)`

        """
        train_X = latin_hypercube_sampling(
            n_sample=n_samples,
            domain_lower_bound=np.array(self.lower_bounds),
            domain_upper_bound=np.array(self.upper_bounds),
        )
        train_y = self.target(train_X)

        return train_X, train_y

    def get_harlow_model(
        self,
        name: str,
    ) -> ForwardModelBase:

        """
        Model factory for the `HarlowSurrogate` class.

        TODO:
            * Refactor this function so that it returns a `ForwardModel` object
                with a `.predict()` method that queries an instance of `Surrogate`.
            * It is important that the generated `ForwardModel` instance does not
                contain the surrogate model itself (due to incompatibility of tensors
                with `copy.deepcopy`), or the problem definition.

        Notes
        ---------
        * This function was originally created as a workaround for passing arguments to
        classes derived from `ForwardModelBase`.

        Parameters
        ----------
        problem
        forward_model
        surrogate
        n_output_dims
        kwargs

        Returns
        -------

        """

        class HarlowSurrogate(ForwardModelBase):

            # Get interface from existing forward model
            def interface(self):
                self.input_sensors = self.kwargs["input_sensors"]
                self.output_sensors = self.kwargs["output_sensors"]
                self.parameters = self.kwargs["parameters"]
                self.input_coordinates = self.kwargs["input_coordinates"]

            def response(self, inp: dict) -> dict:
                # TODO: Currently this method assumes that `Nx` and `Nt` are available.
                # Instead they should be obtained from the problem definition
                params = np.atleast_2d([inp["X" + str(i + 1)] for i in range(4)])
                response = dict()

                # TODO: Adjust `X` to account for the case of sensors with vector
                # input (e.g. timeseries).
                vec_t = self.input_coordinates[self.input_sensors[0].name]

                # Evaluate function and arange output on grid
                f = self.kwargs["func_pred"](params, return_std=False)
                f = np.reshape(f, (len(self.output_sensors), len(vec_t)))

                for idx_x, os in enumerate(self.output_sensors):
                    response[os.name] = f[idx_x, :]

                return response

        return HarlowSurrogate(
            name,
            func_pred=self.func_pred,
            input_sensors=self.input_sensors,
            output_sensors=self.output_sensors,
            parameters=self.parameters,
            input_coordinates=self.input_sensor_data,
        )

    # TODO: This will likely fail for multidimensional parameters. FIX!
    def _get_bounds(self):

        self.lower_bounds = []
        self.upper_bounds = []
        self.bounds = {}

        # Find the priors corresponding to model parameters
        for key, value in self.problem.priors.items():

            # Error if any of the model priors are not uniform
            if value.prior_type != "uniform":
                raise ValueError(
                    f"Non-uniform distribution of type {value.prior_type}"
                    f" found for prior {key}. The `HarlowSurrogate` currently"
                    f" only supports uniform priors."
                )

        # Dictionary of upper and lower bounds
        for param in self.parameters:
            lb = self.problem.constant_prms_dict["low_" + param]
            ub = self.problem.constant_prms_dict["high_" + param]

            # List of upper and lower bounds
            self.lower_bounds.append(lb)
            self.upper_bounds.append(ub)

            # Append to dict
            self.bounds[param] = {"low": lb, "high": ub}

    def _cast_to_numpy(self, inp: dict) -> np.ndarray:
        raise NotImplementedError

    def _cast_to_dict(self, X: np.ndarray) -> dict:
        return dict(zip(self.parameters, X))

    def target(self, X: np.ndarray) -> np.ndarray:
        # TODO: Adjust `y` to account for the case of sensors with vector
        # input (e.g. timeseries).

        # Initialize output array
        y = np.zeros((X.shape[0], len(self.output_sensors) * len(self.vec_t)))

        # Evaluate physical model for each input vector
        for idx_Xi, Xi in enumerate(X):
            response = self.forward_model.response(self._cast_to_dict(Xi))
            sensor_out = []
            for idx_os, os in enumerate(self.output_sensors):
                sensor_out.append(response[os.name])
            y[idx_Xi, :] = np.ravel(sensor_out)

        return np.array(y)
