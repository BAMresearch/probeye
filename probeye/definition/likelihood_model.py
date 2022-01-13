# standard library
from typing import Union, List, Optional

# third party imports
import numpy as np

# local imports
from probeye.definition.sensor import Sensor
from probeye.subroutines import len_or_one
from probeye.subroutines import make_list
from probeye.subroutines import translate_prms_def
from probeye.subroutines import get_dictionary_depth


class GaussianLikelihoodModel:
    """
    This class describes a Gaussian (i.e., normal) likelihood model in general terms.
    It contains information such as the likelihood model's latent parameters, its scope
    with respect to the given experiments, the sensors it considers, its error model
    specification as well as its correlation structure.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union[Sensor, List[Sensor]],
        experiment_names: Union[str, List[str], None] = None,
        additive_model_error: bool = True,
        multiplicative_model_error: bool = False,
        additive_measurement_error: bool = False,
        correlation_variables: str = "",
        correlation_model: str = "exp",
        correlation_dict: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        prms_def
            Parameter names defining which parameters are used by the likelihood model.
            For example prms_def = ['mu', 'sigma']. To check out the other possible
            formats, see the explanation for the same parameter in probeye/definition/
            forward_model.py:ForwardModelBase.__init__.
        sensors
            These are the sensor objects which serve as output sensors in one of the
            problem's forward models, that the likelihood model should refer to. This
            means, the likelihood model should describe the model error between the
            model response for all sensors specified in this 'sensors'-argument and the
            corresponding experimental data.
        experiment_names
            Defines the experiments the likelihood model should refer to. This means
            that the likelihood model will describe the error of the model response with
            respect to the specified experiments here. If the value is None, the
            experiments will be derived automatically by finding all of the problem's
            experiments that contain all of the sensor's names (meaning the name
            attribute of the sensor-objects specified in the 'sensors'-argument above)
            as sensor_values. For more information on this automatic assignment, check
            out assign_experiments_to_likelihood_models in probeye/definition/
            inference_problem.py.
        additive_model_error
            If True, the model error is assumed to be additive and not multiplicative.
            Note that in this case 'multiplicative_model_error' must be False.
        multiplicative_model_error
            If True, the model error is assumed to be multiplicative and not additive.
            Note that in this case 'additive_model_error' must be False.
        additive_measurement_error
            If True, next to the model error, a normal, zero-mean i.i.d. measurement
            error is assumed to be present.
        correlation_variables
            Defines the correlation variables. This argument can be any combination of
            the characters 'x', 'y', 'z', 't', each one appearing at most once. Examples
            are: 'x', 't', 'xy', 'yzt'.
        correlation_model
            Defines the correlation function to be used in case correlation is
            considered (which is the case, when correlation_variables is a non-empty
            string). Currently, there is only one option 'exp' which represents an
            exponential model. In the future, more options should be added.
        correlation_dict
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
        name
            Unique name of the likelihood model. This name is None, if the user does not
            specify it when adding the likelihood model to the problem. It is then named
            automatically before starting the inference engine.
        """

        # general attributes
        self.name = name
        self.prms_def, self.prms_dim = translate_prms_def(prms_def)

        # attributes related to the error model
        self.additive_model_error = additive_model_error
        self.multiplicative_model_error = multiplicative_model_error
        self.additive_measurement_error = additive_measurement_error

        # sensor-related attributes
        self.sensors = make_list(sensors)
        self.sensor_names = [sensor.name for sensor in self.sensors]
        self.n_sensors = len(self.sensor_names)

        # correlation-related attributes
        self.correlation_variables = correlation_variables
        self.correlation_model = correlation_model
        self.correlation_dict = correlation_dict
        self.considers_correlation = True if self.correlation_variables else False
        self.considers_time_correlation = "t" in self.correlation_variables
        self.considers_space_correlation = (
            "x" in self.correlation_variables
            or "y" in self.correlation_variables
            or "z" in self.correlation_variables
        )
        self.considers_only_time_correlation = (
            self.considers_time_correlation and not self.considers_space_correlation
        )
        self.considers_only_space_correlation = (
            self.considers_space_correlation and not self.considers_time_correlation
        )
        self.considers_space_and_time_correlation = (
            self.considers_time_correlation and self.considers_space_correlation
        )
        self.considers_time_and_space_correlation = (
            self.considers_space_and_time_correlation
        )

        # add the experiment_names to the log-likelihood model; additionally, the flag
        # is required, so that the experiment_names are not overwritten by the
        # automatic routine InferenceProblem.assign_experiments_to_likelihood_models
        if experiment_names is not None:
            self.experiment_names = make_list(experiment_names)
            self.assign_experiments_automatically = False
        else:
            self.experiment_names = []
            self.assign_experiments_automatically = True

        # as soon as defined, this attribute will be a pointer to the inference
        # problems experiments (it will be used for consistency checks)
        self.problem_experiments = {}  # type: dict

    @property
    def n_experiments(self) -> int:
        """
        Provides a dynamic attributes stating the number of experiments that
        were assigned to the log-likelihood model.
        """
        return len(self.experiment_names)

    def check_correlation_definition(self, valid_corr_models: tuple = ("exp",)):
        """
        Check if the correlation definition is valid.

        Parameters
        ----------
        valid_corr_models
            The tuple contains all currently implemented correlation models.
        """

        # check that only valid characters are given, and that those characters
        # are at most mentioned once
        for char in self.correlation_variables:
            if char not in ["x", "y", "z", "t"]:
                raise RuntimeError(
                    f"Found invalid correlation variable '{char}' in the correlation "
                    f"definition. Only the characters 'x', 'y', 'z', 't' are valid "
                    f"correlation variables."
                )
            if self.correlation_variables.count(char) > 1:
                raise RuntimeError(
                    f"The correlation variable '{char}' was mentioned more than once "
                    f"in the correlation definition: '{self.correlation_variables}'."
                )

        # check the correlation model
        if self.correlation_model not in valid_corr_models:
            raise ValueError(
                f"Found invalid correlation model '{self.correlation_model}' in the "
                f"correlation definition. Currently, valid correlation models "
                f"are: {valid_corr_models}."
            )

        # check if the model error definition is valid
        if self.additive_model_error and self.multiplicative_model_error:
            raise RuntimeError(
                f"It is not possible to consider both an additive and a multiplicative "
                f"model error at the same time. Please unselect one of them."
            )

    def add_experiments(self, experiment_names_: Union[str, List[str]]):
        """
        Adds experiment names to the log-likelihood model. When the log-likelihood model
        is evaluated it will only be evaluated for those experiments added here.

        Parameters
        ----------
        experiment_names_
            Names (strings) of experiments from the InferenceProblem that should be
            added to the log-likelihood model.
        """
        # check if the given experiments are compatible with the log-likelihood model
        # with respect to the sensors
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
                        f"log-likelihood model."
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
                    f"The experiment '{exp_name}' has already been added to this "
                    f"likelihood model. Something might be wrong here."
                )
        self.experiment_names += experiment_names

    def prepare_corr_dict(self):
        """
        Ensures that the corr_dict attribute is a depth=2 dictionary where the first
        keys are the likelihood model's experiment names, and the second set of keys
        correspond to the correlation variables.
        """
        # check the correlation dictionary (corr_dict); note that this dict
        # can have two different structures, see the explanation in __init__
        if self.correlation_dict is None:
            self.correlation_dict = {"x": "x", "y": "y", "z": "z", "t": "t"}
        depth = get_dictionary_depth(self.correlation_dict)
        if depth == 2:
            # in this case, self.corr_dict describes an experiment-wise mapping
            for exp_name, exp_corr_dict in self.correlation_dict.items():
                for char in self.correlation_variables:
                    if char not in exp_corr_dict:
                        raise KeyError(
                            f"The specified correlation variable '{char}' is "
                            f"missing in the give corr_dict for experiment "
                            f"'{exp_name}'."
                        )
        elif depth == 1:
            # in this case, self.corr_dict describes a global mapping
            for char in self.correlation_variables:
                if char not in self.correlation_dict:
                    raise KeyError(
                        f"The specified correlation variable '{char}' is "
                        f"missing in the give corr_dict."
                    )
            # translate the global correlation dictionary to the experiment-wise
            # format, so that in each of the two cases described here, it has
            # the same format (required for later processing)
            correlation_dict = {}
            for exp_name in self.experiment_names:
                correlation_dict[exp_name] = self.correlation_dict
            self.correlation_dict = correlation_dict

    def residuals(self, model_response_dict: dict) -> dict:
        """
        Computes the model residuals (model prediction minus measurement) for all of the
        likelihood model's experiments and returns them as a dictionary.

        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.

        Returns
        -------
        residuals_dict
            A dictionary with the keys being the likelihood model's sensor names, and 1D
            numpy arrays representing the model residuals for all experiments as values.
        """
        # prepare the dictionary keys
        residuals_dict = {name: np.array([]) for name in self.sensor_names}

        # fill the dictionary with model residual vectors
        for exp_name in self.experiment_names:
            exp_dict = self.problem_experiments[exp_name]
            ym_dict = model_response_dict[exp_name]
            ye_dict = exp_dict["sensor_values"]
            residuals_dict = {
                name: np.append(residuals_dict[name], ym_dict[name] - ye_dict[name])
                for name in self.sensor_names
            }

        return residuals_dict

    def residuals_vector(self, model_response_dict: dict) -> np.ndarray:
        """
        Computes the model residuals for all of the likelihood model's sensors over all
        of the likelihood model's experiments and returns them in a single vector.

        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.

        Returns
        -------
        residuals_vector
            A one-dimensional vector containing the model residuals.
        """
        residuals_dict = self.residuals(model_response_dict)
        n = 0
        for residuals_sub_vector in residuals_dict.values():
            n += len_or_one(residuals_sub_vector)
        residuals_vector = np.zeros(n)
        idx = 0
        for residuals_sub_vector in residuals_dict.values():
            m = len_or_one(residuals_sub_vector)
            residuals_vector[idx : idx + m] = residuals_sub_vector
            idx += m
        return residuals_vector

    def coordinate_vector(self, coord: str) -> np.ndarray:
        """

        Parameters
        ----------
        coord
            A single character from {'x', 'y', 'z', 't'}, i.e., a possible correlation
            variable.

        Returns
        -------
        coord_vector
            A vector containing the values of the requested coordinate 'coord' over
            all of the likelihood model's experiments and sensors. These values have
            the same structure as the values in the residual vector computed by the
            method 'residuals_vector' above. Hence, the i-th entry in 'coord_vector'
            corresponds to the i-th entry in the vector returned by 'residuals_vector'.
        """

        # check input
        if coord not in ["x", "y", "z", "t"]:
            raise ValueError(
                f"The given coordinate 'coord' is neither 'x', 'y', 'z' nor 't'. "
                f"Found '{coord}'.")

        # prepare the coord-vector with the correct length
        n = 0
        ns = len(self.sensors)
        for exp_name in self.experiment_names:
            exp_sensor_values = self.problem_experiments[exp_name]["sensor_values"]
            coord_name_in_exp = self.correlation_dict[exp_name][coord]
            if coord_name_in_exp in exp_sensor_values:
                n += len_or_one(exp_sensor_values[coord_name_in_exp]) * ns
            else:
                for sensor in self.sensors:
                    try:
                        n += len_or_one(getattr(sensor, coord))
                    except AttributeError:
                        print(f"Sensor '{sensor.name}' of likelihood model "
                              f"'{self.name}' does not have a '{coord}'-attribute!")
                        raise
        coord_vector = np.zeros(n)

        # fill the dictionary with model residual vectors
        i = 0
        for exp_name in self.experiment_names:
            exp_sensor_values = self.problem_experiments[exp_name]["sensor_values"]
            coord_name_in_exp = self.correlation_dict[exp_name][coord]
            if coord_name_in_exp in exp_sensor_values:
                coord_sub_vector = exp_sensor_values[coord_name_in_exp]
                m = len_or_one(coord_sub_vector)
                for _ in range(ns):
                    coord_vector[i: i + m] = coord_sub_vector
                    i += m
            else:
                for sensor in self.sensors:
                    coord_sub_vector = getattr(sensor, coord)
                    m = len_or_one(coord_sub_vector)
                    coord_vector[i: i + m] = coord_sub_vector
                    i += m

        return coord_vector
