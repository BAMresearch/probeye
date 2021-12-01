# standard library
from typing import Union, List, Optional

# third party imports
import numpy as np

# local imports
from probeye.definition.sensor import Sensor
from probeye.subroutines import make_list
from probeye.subroutines import len_or_one
from probeye.subroutines import translate_prms_def
from probeye.subroutines import get_dictionary_depth


class NoiseModelBase:
    """
    General noise model without any (solver-specific) computing methods.
    """

    def __init__(
        self,
        dist: str,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union[Sensor, List[Sensor]],
        experiment_names: Union[str, List[str], None] = None,
        name: Optional[str] = None,
        corr_static: str = "",
        corr_dynamic: str = "",
        corr_model: str = "exp",
        corr_dict: Optional[dict] = None,
        noise_type: str = "additive",
    ):
        """
        Parameters
        ----------
        dist
            A string specifying the probability distribution the noise model is based
            on, e.g. 'normal' or 'uniform'. This string will be used to translate an
            object of this general noise model to a solver-specific one. To check out
            which values are valid for the solver you want to use, take a look at the
            translate_noise_model-method in probeye/inference/scipy_/noise_models.py
            for scipy and emcee and in probeye/inference/torch_/noise_models.py for
            the pyro-solver.
        prms_def
            Parameter names (strings) defining which parameters are used by the noise
            model. E.g. prms_def = ['mu', 'sigma']. To check out the other possible
            formats, see the explanation for the same parameter in probeye/definition/
            forward_model.py:ForwardModelBase.__init__.
        sensors
            These are sensor objects which serve as output sensors in one of the
            problem's forward models, that the noise model should refer to. This means,
            the noise model should describe the model error between the model response
            for all sensors specified in this 'sensors'-argument and the corresponding
            experimental data.
        experiment_names
            Defines the experiments that the noise model should refer to. This means
            that the noise model will describe the error of the model  response with
            respect to the specified experiments here. If the value is None, the
            experiments will be derived automatically by finding all of the problem's
            experiments that contain all of the sensor's names (meaning the name
            attribute of the sensor-objects specified in the 'sensors'-argument above)
            as sensor_values. For more information on this automatic assignment, check
            out the assign_experiments_to_noise_models-method in probeye/definition/
            inference_problem.py.
        name
            Unique name of the noise model. This name is None, if the user does not
            specify it when adding the noise model to the problem. It is then named
            automatically before starting the inference engine.
        corr_static
            Defines the static correlation data. Static correlation data is experimental
            data that is constant over all of the noise model's experiments (think of
            the fixed positions of some strain gauges for example). This data is to be
            found as attributes of the noise model's sensors. The corr_static argument
            can be any combination of the characters 'x', 'y', 'z', 't', each one
            appearing at most once. Examples are: 'x', 't', 'xy', 'yzt'.
        corr_dynamic
            Defines the dynamic correlation data. Dynamic correlation data is
            experimental data that can (and usually will) change between the noise
            model's experiments. This data is to be found as in the sensor_values of the
            noise model's experiments. The corr_static argument can be any combination
            can assume the same values as the corr_static argument. However, each of the
            variables 'x', 'y', 'z', 't' can only appear at most in one of the two
            arguments. So, for example corr_static='xt' and corr_dynamic='xz' would not
            be valid since 'x' appears in both strings.
        corr_model
            Defines the correlation function to be used in case correlation is
            considered (which is the case, when at least one of corr_static and
            corr_dynamic is a non-empty string). Currently, there is only one option
            'exp' which represents an exponential model. In the future, more options
            should be added.
        corr_dict
            Allows to give each correlation variable ('x', 'y', 'z', 't') an alias used
            in the actual experimental data. This dictionary can either provide a
            mapping that holds for all experiments, or a mapping that is individual for
            each experiment. In the first case, you would have something like
            {'x': 'lateral_position_1', 't': 'T'}. Here, in all experiments the
            sensor_value with the key 'lateral_ position_1' would be understood as the
            spatial correlation variable 'x' and the sensor_value with key 'T' would be
            understood as the time 't'. In the second case, when the mapping is
            different for each experiment, corr_dict would have the experiment names as
            keys, and the values would be dictionaries, as explained in the first case.
            An example would be {'Exp_Jun23': {'x': 'x_23', 't': 't_23'},
            'Exp_Jun93': {'x': 'x_29', 't': 't_29'}}. If corr_dict is None, it is
            assumed that the correlation variables have the same names in the sensor_
            values of the experiments as the default values used here, i.e., 'x', 'y',
            'z', 't'.
        noise_type
            Either 'additive', 'multiplicative' or 'other'. Defines if the error is
            computed via [prediction - measurement] ('additive') or via
            [prediction/measurement-1] ('multiplicative') or in some 'other' i.e.,
            non-standard fashion.
        """

        # general attributes
        self.dist = dist
        self.prms_def, self.prms_dim = translate_prms_def(prms_def)
        self.name = name
        self.noise_type = noise_type

        # sensor-related attributes
        self.sensors = make_list(sensors)
        self.sensor_names = [sensor.name for sensor in self.sensors]
        self.n_sensors = len(self.sensor_names)

        # add the experiment_names to the noise model; additionally, the flag is
        # required, so that the experiment_names are not overwritten by the
        # automatic routine InferenceProblem.assign_experiments_to_noise_models
        if experiment_names is not None:
            self.experiment_names = make_list(experiment_names)
            self.assign_experiments_automatically = False
        else:
            self.experiment_names = []
            self.assign_experiments_automatically = True

        # correlation-related attributes
        self.corr_static = corr_static
        self.corr_dynamic = corr_dynamic
        self.corr = self.corr_static + self.corr_dynamic
        self.corr_model = corr_model
        self.corr_dict = corr_dict
        self.check_correlation_definition()

        # as soon as defined, this attribute will be a pointer to the inference
        # problems experiments (it will be used for consistency checks)
        self.problem_experiments = {}  # type: dict

        # set the error_function depending on the noise-type
        if noise_type == "additive":
            self.error_function = self.error_function_additive
        elif noise_type == "multiplicative":
            self.error_function = self.error_function_multiplicative
        elif noise_type == "other":
            self.error_function = self.error_function_other
        else:
            raise ValueError(
                f"Unknown noise_type: '{noise_type}'. The noise_type must be "
                f"either 'additive', 'multiplicative' or 'other'."
            )

    @property
    def n_experiments(self) -> int:
        """
        Provides a dynamic attributes stating the number of experiments that
        were assigned to the noise model.
        """
        return len(self.experiment_names)

    def check_correlation_definition(self, valid_corr_models: tuple = ("exp",)):
        """
        Check if the correlation definition cumulated in self.corr is valid.

        Parameters
        ----------
        valid_corr_models
            The tuple contains all currently implemented correlation models.
        """
        # check that only valid characters are given, and that those characters
        # are at most mentioned once
        for char in self.corr:
            if char not in ["x", "y", "z", "t"]:
                raise RuntimeError(
                    f"Found invalid correlation variable '{char}' in the "
                    f"correlation definition. Only the characters 'x', 'y', "
                    f"'z', 't' are valid correlation variables."
                )
            if self.corr.count(char) > 1:
                raise RuntimeError(
                    f"The correlation variable '{char}' was mentioned more "
                    f"than once in the correlation definition: '{self.corr}'."
                )
        # check the correlation model
        if self.corr_model not in valid_corr_models:
            raise ValueError(
                f"Found invalid correlation model '{self.corr_model}' in the "
                f"correlation definition. Currently, valid correlation models "
                f"are: {valid_corr_models}."
            )

    def prepare_corr_dict(self):
        """
        Ensures that the corr_dict attribute is a depth=2 dictionary where the first
        keys are the noise model's experiment names, and the second set of keys
        correspond to the correlation variables.
        """
        # check the correlation dictionary (corr_dict); note that this dict
        # can have two different structures, see the explanation in __init__
        if self.corr_dict is None:
            self.corr_dict = {"x": "x", "y": "y", "z": "z", "t": "t"}
        depth = get_dictionary_depth(self.corr_dict)
        if depth == 2:
            # in this case, self.corr_dict describes an experiment-wise mapping
            for exp_name, exp_corr_dict in self.corr_dict.items():
                for char in self.corr:
                    if char not in exp_corr_dict:
                        raise KeyError(
                            f"The specified correlation variable '{char}' is "
                            f"missing in the give corr_dict for experiment "
                            f"'{exp_name}'."
                        )
        elif depth == 1:
            # in this case, self.corr_dict describes a global mapping
            for char in self.corr:
                if char not in self.corr_dict:
                    raise KeyError(
                        f"The specified correlation variable '{char}' is "
                        f"missing in the give corr_dict."
                    )
            # translate the global correlation dictionary to the experiment-wise
            # format, so that in each of the two cases described here, it has
            # the same format (required for later processing)
            corr_dict = {}
            for exp_name in self.experiment_names:
                corr_dict[exp_name] = self.corr_dict
            self.corr_dict = corr_dict

    def add_experiments(self, experiment_names_: Union[str, List[str]]):
        """
        Adds experiment names to the noise model. When the noise model is evaluated it
        will only be evaluated for those experiments added here.

        Parameters
        ----------
        experiment_names_
            Names (strings) of experiments from the InferenceProblem that should be
            added to the noise model.
        """
        # check if the given experiments are compatible with the noise model with
        # respect to the sensors
        experiment_names = make_list(experiment_names_)
        forward_models = set()
        for exp_name in experiment_names:
            exp_dict = self.problem_experiments[exp_name]
            forward_models.add(exp_dict["forward_model"])
            sensor_names_exp = [*exp_dict["sensor_values"].keys()]
            for sensor_name in self.sensor_names:
                if sensor_name not in sensor_names_exp:
                    raise RuntimeError(
                        f"Experiment '{exp_name}' does not contain a sensor "
                        f"'{sensor_name}' which is required for the evaluation of the "
                        f"noise model."
                    )
        # check if the given experiments all refer to one forward model
        if len(forward_models) > 1:
            raise RuntimeError(
                f"The given experiments refer to more than one forward model!"
            )
        # check if one of the given experiments have been added before
        for exp_name in experiment_names:
            if exp_name in self.experiment_names:
                raise RuntimeError(
                    f"The experiment '{exp_name}' has already been added to this noise "
                    f"model. Something might be wrong here."
                )
        self.experiment_names += experiment_names

    def error(self, model_response_dict: dict) -> dict:
        """
        Computes the model error for all of the noise model's experiments and returns
        them in a dictionary that is sorted by output sensor_values.

        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.

        Returns
        -------
        model_error
            A dictionary with the keys being the noise model's sensor names, and 1D
            numpy arrays representing the model errors as values.
        """
        # prepare the dictionary keys
        model_error_dict = {name: np.array([]) for name in self.sensor_names}

        # fill the dictionary with model error vectors
        for exp_name in self.experiment_names:
            exp_dict = self.problem_experiments[exp_name]
            ym_dict = model_response_dict[exp_name]
            ye_dict = exp_dict["sensor_values"]
            me_dict = self.error_function(ym_dict, ye_dict)
            model_error_dict = {
                name: np.append(model_error_dict[name], me_dict[name])
                for name in self.sensor_names
            }

        return model_error_dict

    def error_vector(self, model_response_dict: dict) -> np.ndarray:
        """
        Computes the model error for all of the noise model's experiments and returns
        them in a dictionary that is sorted by output sensor_values.

        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.

        Returns
        -------
        error_vector
            A one-dimensional vector containing the model errors.
        """
        model_error_dict = self.error(model_response_dict)
        n = 0
        for error_sub_vector in model_error_dict.values():
            n += len_or_one(error_sub_vector)
        error_vector = np.zeros(n)
        idx = 0
        for error_sub_vector in model_error_dict.values():
            m = len_or_one(error_sub_vector)
            error_vector[idx : idx + m] = error_sub_vector
            idx += m
        return error_vector

    def error_function_additive(self, ym_dict: dict, ye_dict: dict) -> dict:
        """
        Evaluates the additive model error for each of the noise model' sensors.

        Parameters
        ----------
        ym_dict
            The computed values for the model's output sensor_values.
        ye_dict
            The measured values for the model's output sensor_values.

        Returns
        -------
        error_dict
            The computed model error for the model's output sensor_values.
        """
        # for each sensor, its own error metric is used to compute the error
        error_dict = {name: ym_dict[name] - ye_dict[name] for name in self.sensor_names}
        return error_dict

    def error_function_multiplicative(self, ym_dict: dict, ye_dict: dict) -> dict:
        """
        Evaluates the multiplicative model error for each of the noise model's sensors.

        Parameters
        ----------
        ym_dict
            The computed values for the model's output sensor_values.
        ye_dict
            The measured values for the model's output sensor_values.

        Returns
        -------
        error_dict
            The computed model error for the model's output sensor_values.
        """
        # for each sensor, its own error metric is used to compute the error
        error_dict = {
            name: ym_dict[name] / ye_dict[name] - 1.0 for name in self.sensor_names
        }
        return error_dict

    def error_function_other(self, ym_dict: dict, ye_dict: dict) -> dict:
        """
        Non-standard error function self.error_function will point to when self.
        noise_type is set to 'other'. See self.error_function for more information.
        """
        raise NotImplementedError(
            "Your model does not have an non-standard error_function-method yet. If "
            "you want to use it, you need to implement it first."
        )

    def loglike_contribution(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Evaluates the log-likelihood function for the given model error and the given
        noise parameter vector. This method has to be overwritten.

        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.
        prms
            Dictionary containing parameter name:value pairs.
        worst_value
            This value is returned when this method does not result in a numeric value.
            This might happen for example when the given parameters are not valid (for
            example in case of a negative standard deviation). The returned value in
            such cases should represent the worst possible value of the contribution.

        Returns
        -------
        ll
            The evaluated log-likelihood function.
        """
        raise NotImplementedError(
            "Your model does not have a loglike_contribution-method. You need to "
            "define this method so you can evaluate your noise model."
        )


class NormalNoiseModel(NoiseModelBase):
    """
    A general Gaussian (normal) noise model with or without correlations. This class
    provides little additional attributes/methods compared to the NoiseModelBase, but
    since the normal noise model is used very often, the existence of this class results
    in a convenient definition of the normal noise model when setting up the Problem.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union[Sensor, List[Sensor]],
        experiment_names: Union[str, List[str], None] = None,
        name: Optional[str] = None,
        corr_static: str = "",
        corr_dynamic: str = "",
        corr_model: str = "exp",
        corr_dict: Optional[dict] = None,
        noise_type: str = "additive",
    ):
        """
        See docstring of NoiseModelBase for information on the input arguments. Except
        for the missing 'dist'-argument, they are similar.
        """

        # initialize the base class with given input
        super().__init__(
            "normal",
            prms_def,
            sensors,
            experiment_names=experiment_names,
            name=name,
            corr_static=corr_static,
            corr_dynamic=corr_dynamic,
            corr_model=corr_model,
            corr_dict=corr_dict,
            noise_type=noise_type,
        )

        # check that at the standard deviation is provided (this can be either as a
        # constant or a latent parameter, but it has to be given); note that 'std' has
        # to be used as the local name
        if "std" not in [*self.prms_def.values()]:
            raise RuntimeError(
                "The standard deviation 'std' was not provided in prms_def!"
            )

        # the mean value(s) do not have to be stated explicitly; if they are not given,
        # the are assumed to be zero
        self.zero_mean = True
        if "mean" in [*self.prms_def.values()]:
            self.zero_mean = False
