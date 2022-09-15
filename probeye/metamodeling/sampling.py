# standard library
from typing import Tuple, Callable
import copy

# third party imports
import numpy as np
import pandas as pd
from scipy.stats import qmc

# local imports
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.inference.scipy.priors import translate_prior
from probeye.subroutines import len_or_one

# external imports
from harlow.sampling import Sampler as HarlowSamplerBase
from harlow.surrogating import Surrogate as HarlowSurrogateBase
from harlow.utils.helper_functions import latin_hypercube_sampling


class SamplerBase:
    """
    Base class for probeye samplers
    """

    def __init__(self, problem: InverseProblem):
        # the considered inverse problem
        self.problem = problem

        # the sampling happens before the problem is given to a solver; this means that
        # the priors of the problem are merely descriptive and they have to be
        # translated to have their respective computing routines
        self.priors = copy.deepcopy(self.problem.priors)
        for prior_template in self.problem.priors.values():
            prm_name = prior_template.ref_prm
            self.priors[prm_name] = translate_prior(prior_template)

    def sample(
        self, forward_model: ForwardModelBase, **kwargs
    ) -> Tuple[pd.DataFrame, dict]:
        raise NotImplementedError


class LatinHypercubeSampler(SamplerBase):
    """
    Contains functionalities to provide samples of an inverse problem's parameters by
    utilizing latin hypercube sampling. The generate samples are intended for training
    of a surrogate model.

    Parameters
    ----------
    problem
        The considered inverse problem.
    """

    def __init__(self, problem):
        super().__init__(problem)

    def generate_samples(self, n_samples: int, seed: int = 1) -> np.ndarray:
        """
        Samples the problems latent parameter vector from the parameter's priors in
        combination with latin hypercube sampling.

        Parameters
        ----------
        n_samples
            The number of requested samples.
        seed
            Seed for random number generator.

        Returns
        -------
        sample_array
            The sampled latent parameter vectors. Each row corresponds to a single
            latent parameter vector.
        """

        # make sure that all parameters are one-dimensional; it is not straight forward
        # how to do LHS for general multivariate parameters
        for prm_name in self.problem.latent_prms:
            if self.problem.parameters[prm_name].dim > 1:
                raise RuntimeError(
                    f"The given problem has a multivariate parameter ('{prm_name}') "
                    f"for which LHS is not supported."
                )

        # the dimension of the required hypercube is the number of parameters; note
        # that this only holds since all parameters are 1D (see the check above)
        dim = self.problem.n_latent_prms

        # create the requested number of latin hypercube samples with the requested dim.
        lhd = qmc.LatinHypercube(d=dim, seed=seed).random(n=n_samples)

        # this is going to be the array of the parameter samples; each row will
        # correspond to a theta-vector
        sample_array = np.zeros(lhd.shape)

        # fill the prepared array for the parameter samples using the LHS samples
        for prm_name, parameter in self.problem.parameters.items():
            if parameter.is_latent:
                idx = parameter.index
                prior = self.priors[prm_name]
                prms = self.problem.get_constants(prior.hyperparameters)
                for lt_prm in self.problem.get_latent_prior_hyperparameters(prm_name):
                    idx_lt = self.problem.parameters[lt_prm].index
                    prms[lt_prm] = sample_array[:, idx_lt]
                q = lhd[:, idx]
                sample_array[:, idx] = prior(prms, "ppf", q, use_ref_prm=False)

        return sample_array

    def sample(
        self,
        forward_model: ForwardModelBase,
        n_samples: int,
        seed: int = 1,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Generates a given number of training data for fitting a surrogate model. The
        training data contains a number of parameter vectors (sampled using LHS) and
        the corresponding model responses.

        Parameters
        ----------
        forward_model
            The forward model that should be evaluated.
        n_samples
            The number of parameter vectors the forward model should be evaluated for.
        seed
            Seed for random number generator.

        Returns
        -------
        prm_samples_pd
            The parameter samples the forward model was evaluated at.
        responses_over_experiments
            The keys are the names of the experiment the forward model is associated
            with, while the values are 3D-arrays containing the forward model's
            response. responses_over_experiments[i][j] will contain the forward  model's
            response with the ith parameter-vector for the jth output sensor.
        """

        # get the forward model object with the given name and prepare the corresponding
        # experimental in- and output dictionaries
        forward_model.prepare_experimental_inputs_and_outputs()

        # generate the latent parameter samples and convert it to a data frame to have
        # the association between columns and parameter names
        prm_samples = self.generate_samples(n_samples, seed=seed)
        prm_samples_pd = pd.DataFrame(
            prm_samples, columns=self.problem.get_theta_names()
        )

        # this dictionary will contain the forward model responses for each of the
        # experiments associated with the forward model; so the keys will be experiment
        # names while the values will be 3D-arrays with the forward model's responses;
        # responses_over_experiments[i] will correspond to the response of the ith
        # parameter vector; responses_over_experiments[i][j] will contain the forward
        # model's response with the ith parameter vector for the jth output sensor
        responses_over_experiments = {}

        # here, the length of the vectors of the forward model's output sensors is
        # determined; to that end, the forward model is evaluated once
        first_exp_name = forward_model.experiment_names[0]
        exp_inp = forward_model.input_from_experiments[first_exp_name]
        first_theta = prm_samples[0]
        prms_model = self.problem.get_parameters(first_theta, forward_model.prms_def)
        inp = {**exp_inp, **prms_model}
        response_dict = forward_model.response(inp)
        # make sure that the vectors returned by each of the forward model's output
        # sensors has the same length; otherwise an AssertionError is raised
        length_set = set()
        for value in response_dict.values():
            length_set.add(len_or_one(value))
        assert len(length_set) == 1
        n_out_values = list(length_set)[0]

        # evaluate the forward model for each experiment/parameter vector
        for exp_name in forward_model.experiment_names:
            exp_inp = forward_model.input_from_experiments[exp_name]
            response_array = np.zeros(
                (n_samples, forward_model.n_output_sensors, n_out_values)
            )
            for i, theta in enumerate(prm_samples):
                prms_model = self.problem.get_parameters(theta, forward_model.prms_def)
                inp = {**exp_inp, **prms_model}  # adds the two dictionaries
                response_dict = forward_model.response(inp)
                for j, response_vector in enumerate(response_dict.values()):
                    response_array[i, j, :] = response_vector
            responses_over_experiments[exp_name] = response_array

        return prm_samples_pd, responses_over_experiments


class HarlowSampler(SamplerBase):
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
        sampler: HarlowSamplerBase,
        surrogate_model: HarlowSurrogateBase,
        fit_points_x: np.ndarray = None,
        fit_points_y: np.ndarray = None,
        test_points_x: np.ndarray = None,
        test_points_y: np.ndarray = None,
        evaluation_metric: Callable = None,
        verbose: bool = False,
        run_name: str = "test",
        **surrogate_kwargs,
    ):

        super().__init__(problem)

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

        # Surrogate model
        self.surrogate = surrogate_model
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
