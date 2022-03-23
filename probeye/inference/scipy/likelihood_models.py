# standard library imports
from typing import Union, Tuple, List, TYPE_CHECKING

# third party imports
import numpy as np
from loguru import logger
from tripy.loglikelihood import chol_loglike_1D
from tripy.loglikelihood import kron_loglike_2D_tridiag
from tripy.loglikelihood import kron_loglike_2D
from tripy.loglikelihood import chol_loglike_2D
from tripy.loglikelihood import _loglike_multivariate_normal
from tripy.utils import inv_cov_vec_1D
from tripy.utils import correlation_matrix
from tripy.utils import correlation_function

# local imports
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.subroutines import len_or_one

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.sensor import Sensor


class ScipyLikelihoodBase(GaussianLikelihoodModel):
    """
    This class serves as a parent class for all the scipy-based likelihood model classes
    that follow below. All of these classes have in common that they contain computation
    methods based on numpy/scipy. In particular, these methods use numpy.ndarrays (in
    contrast to torch.Tensor objects as in pyro).

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

        # this attribute is not set in the __init__ of GaussianLikelihoodModel
        self.problem_experiments = problem_experiments

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes and returns the log-likelihood. To be overwritten by one of the
        specific classes listed below.

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
            A scalar value representing the evaluated log-likelihood function.
        """
        raise NotImplementedError(
            "The ScipyLikelihoodBase class does not have a loglike implementation."
        )


class UncorrelatedModelError(ScipyLikelihoodBase):
    """
    This class serves as a parent class for the two scipy-based likelihood models that
    do not account for correlation. These model-classes are:
    AdditiveUncorrelatedModelError and MultiplicativeUncorrelatedModelError.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

        # this attribute states the number of elements in the likelihood model's
        # response vector (i.e., the stacked vector of the forward model response vector
        # over all of the likelihood model's experiments); this number is defined via
        # the number of experiments defined for the likelihood model as well as the
        # number of elements in a single model response (a forward model evaluation);
        # this attribute is set as soon as the first stacked response vector is computed
        self.n_stacked_response_vector = 0

    def stacked_residuals_vector(self, model_response_dict: dict) -> np.ndarray:
        """
        Computes the model residuals for all of the likelihood model's sensors over all
        of the likelihood model's experiments and returns them in one stacked vector.

        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.

        Returns
        -------
        residuals_vector
            A one-dimensional vector containing the stacked model residuals. The length
            of this vector is given by self.n_stacked_response_vector.
        """

        # in the first call, self.n_stacked_response_vector is not set yet (it's zero),
        # hence it is derived; for calls after that first call, the attribute is set
        # and the two for-loops in the following if-clause are skipped
        if self.n_stacked_response_vector == 0:
            for exp_name in self.experiment_names:
                ye_dict = self.problem_experiments[exp_name]["sensor_values"]
                for sensor_name in self.sensor_names:
                    self.n_stacked_response_vector += len_or_one(ye_dict[sensor_name])

        # allocate the vector of stacked residuals with the correct length
        residuals_vector = np.zeros(self.n_stacked_response_vector)

        # fill the residuals vector; note that the values in the vector are structured
        # like: [sensor1_exp_1, sensor1_exp_2, ..., sensor2_exp_1, sensor2_exp_2, ...],
        # where the order of the sensors is defined by the order in self.sensors and the
        # order of the experiments is defined via self.experiment_names
        m = 0
        for exp_name in self.experiment_names:
            ye_dict = self.problem_experiments[exp_name]["sensor_values"]
            ym_dict = model_response_dict[exp_name]
            for sensor_name in self.sensor_names:
                n = len_or_one(ye_dict[sensor_name])
                residuals_vector[m : m + n] = (
                    ym_dict[sensor_name] - ye_dict[sensor_name]
                )
                m += n

        return residuals_vector

    def stacked_responses_and_residuals_vector(
        self, model_response_dict: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Similar to self.stacked_residuals_vector but this method additionally returns
        the stacked vector of model responses.

        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.

        Returns
        -------
            responses_vector
                A one-dimensional vector containing the stacked model responses. The
                 length of this vector is given by self.n_stacked_response_vector.
            residuals_vector
                A one-dimensional vector containing the stacked model residuals. The
                 length of this vector is given by self.n_stacked_response_vector.
        """

        # in the first call, self.n_stacked_response_vector is not set yet (it's zero),
        # hence it is derived; for calls after that first call, the attribute is set
        # and the two for-loops in the following if-clause are skipped
        if self.n_stacked_response_vector == 0:
            for exp_name in self.experiment_names:
                ye_dict = self.problem_experiments[exp_name]["sensor_values"]
                for sensor_name in self.sensor_names:
                    self.n_stacked_response_vector += len_or_one(ye_dict[sensor_name])

        # allocate the vector of stacked responses/residuals with the correct length
        responses_vector = np.zeros(self.n_stacked_response_vector)
        residuals_vector = np.zeros(self.n_stacked_response_vector)

        # fill the two vectors; note that the values in the vectors are structured
        # like: [sensor1_exp_1, sensor1_exp_2, ..., sensor2_exp_1, sensor2_exp_2, ...],
        # where the order of the sensors is defined by the order in self.sensors and the
        # order of the experiments is defined via self.experiment_names
        m = 0
        for exp_name in self.experiment_names:
            ye_dict = self.problem_experiments[exp_name]["sensor_values"]
            ym_dict = model_response_dict[exp_name]
            for sensor_name in self.sensor_names:
                n = len_or_one(ye_dict[sensor_name])
                responses_vector[m : m + n] = ym_dict[sensor_name]
                residuals_vector[m : m + n] = (
                    ym_dict[sensor_name] - ye_dict[sensor_name]
                )
                m += n

        return responses_vector, residuals_vector


class CorrelatedModelError(ScipyLikelihoodBase):
    """
    This class serves as a parent class for the two scipy-based likelihood models that
    do account for correlation. These model-classes are: SpaceOrTimeCorrelatedModelError
    and SpaceAndTimeCorrelatedModelError.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

        # check the consistency of the correlation definition
        (
            self.space_given_as_vector,
            self.time_given_as_vector,
        ) = self.check_correlation_info_across_experiments()

    def check_correlation_info_across_experiments(self) -> Tuple[bool, bool]:
        """
        Checks the correlation_info dictionaries of all of the likelihood model's
        experiments on their inter-experiment consistency.

        Returns
        -------
        space_given_as_vector
            True if the spatial correlation variables are given in vector format
            throughout the likelihood model's experiments. False if the spatial
            correlation variables are given in scalar format throughout the likelihood
            model's experiments, or when there are no spatial correlation variables.
        time_given_as_vector
            True if the time correlation variable is given in vector format throughout
            the likelihood model's experiments. False if the time correlation variable
            is given in scalar format throughout the likelihood model's experiments, or
            when there is no time correlation variable.
        """

        # check the entries in the correlation_info dicts only
        for correlation_variable in self.correlation_variables:
            for exp_name in self.experiment_names:
                corr_info = self.problem_experiments[exp_name]["correlation_info"]
                for out_sensor_name, inner_dict in corr_info.items():
                    key_list = [*inner_dict.keys()]
                    if correlation_variable not in key_list:
                        raise RuntimeError(
                            f"The correlation_info of experiment '{exp_name}' of "
                            f"likelihood model '{self.name}' is lacking the "
                            f"association of the general correlation variable "
                            f"'{correlation_variable}' to the likelihood model's "
                            f"sensor '{out_sensor_name}'."
                        )

        # derive a dictionary that describes which correlation variables are given as
        # vectors and which are not (i.e., which are given as scalars); such a dict
        # could look like {'y1': {'x': False, 't': True}, 'y2': {'x': False, 't': True}}
        is_vector_dict = {}  # type: dict
        for sensor_name in self.sensor_names:
            is_vector_dict[sensor_name] = {}
            for correlation_variable in self.correlation_variables:
                is_vector_dict[sensor_name][correlation_variable] = True
                for exp_name in self.experiment_names:
                    sensor_values = self.problem_experiments[exp_name]["sensor_values"]
                    corr_info = self.problem_experiments[exp_name]["correlation_info"]
                    local_name = corr_info[sensor_name][correlation_variable]
                    if len_or_one(sensor_values[local_name]) == 1:
                        is_vector_dict[sensor_name][correlation_variable] = False
            sensors_bool_list = [*is_vector_dict[sensor_name].values()]
            if sensors_bool_list.count(True) == 0:
                raise RuntimeError(
                    f"No correlation variable is given as a vector throughout the "
                    f"experiments of likelihood model '{self.name}'"
                )
            if sensors_bool_list.count(True) > 1:
                if "t" in is_vector_dict[sensor_name]:
                    if is_vector_dict[sensor_name]["t"]:
                        raise RuntimeError(
                            f"Found both a time and a space correlation variable given "
                            f"as a vector in the experiments of likelihood model"
                            f"'{self.name}'."
                        )
        bool_dicts = [*is_vector_dict.values()]
        bool_dict_1 = bool_dicts[0]
        for bool_dict in bool_dicts[1:]:
            if bool_dict != bool_dict_1:
                raise RuntimeError(
                    f"The vector-scalar structure of the correlation variables differs "
                    f"across the likelihood model's ({self.name}) sensors."
                )
        # after this check it suffices to consider bool_dict_1 only
        time_given_as_vector = False
        if "t" in bool_dict_1:
            time_given_as_vector = bool_dict_1["t"]
        space_booleans = set()
        for correlation_variable in self.correlation_variables:
            if correlation_variable != "t":
                space_booleans.add(bool_dict_1[correlation_variable])
        if len(space_booleans) == 0:
            space_given_as_vector = False
        elif len(space_booleans) == 1:
            space_given_as_vector = list(space_booleans)[0]
        else:  # len(space_booleans) > 1
            raise RuntimeError(
                f"Found mixed scalar/vector definitions for the spatial correlation "
                f"variables in likelihood model '{self.name}'."
            )

        return space_given_as_vector, time_given_as_vector

    def unionize_correlation_variable(self, correlation_variable: str) -> np.ndarray:
        """
        Goes over all of the likelihood model's experiments and derives the sorted
        union of the sensor_values associated with the given correlation_variable.
        If not all experiments share the same data for the given correlation_variable,
        the old sensor_values values in each experiment will be overwritten with the
        unionized values, and the values of the likelihood model's sensors will be
        interpolated at the values of the unionized sensor values.

        Parameters
        ----------
        correlation_variable
            A default correlation variable character, i.e., 'x', 'y', 'z' or 't'.

        Returns
        -------
        union_vector
            The unionized and sorted sensor_values.
        """

        # the given correlation_variable must be a default one
        if correlation_variable not in ["x", "y", "z", "t"]:
            raise ValueError(
                f"Invalid default correlation variable! Found '{correlation_variable}' "
                f"while ony 'x', 'y', 'z' or 't' are permitted."
            )

        # initialize the union_vector from the first experiment
        union_set = set()  # type: set
        differences_across_lm_sensors = -1
        first_exp_name = self.experiment_names[0]
        for lm_sensor_name in self.sensor_names:
            sensor_name = self.problem_experiments[first_exp_name]["correlation_info"][
                lm_sensor_name
            ][correlation_variable]
            values = self.problem_experiments[first_exp_name]["sensor_values"][
                sensor_name
            ]
            if len_or_one(values) == 1:
                raise RuntimeError(
                    f"Unionization is intended for vector-valued correlation variables."
                    f" However, the given correlation variable '{correlation_variable}'"
                    f" is scalar-valued for the likelihood model's sensor "
                    f"'{lm_sensor_name}' in experiment '{first_exp_name}'"
                )
            union_set_i = set(values)
            if union_set != union_set_i:
                differences_across_lm_sensors += 1
                union_set = union_set.union(union_set_i)
        shared_by_all = True
        if differences_across_lm_sensors > 0:
            shared_by_all = False

        # check/extend shared vector over remaining experiments
        for exp_name in self.experiment_names[1:]:
            for lm_sensor_name in self.sensor_names:
                sensor_name = self.problem_experiments[exp_name]["correlation_info"][
                    lm_sensor_name
                ][correlation_variable]
                values = self.problem_experiments[exp_name]["sensor_values"][
                    sensor_name
                ]
                union_set_i = set(values)
                if union_set != union_set_i:
                    shared_by_all = False
                    union_set = union_set.union(union_set_i)

        # sorting is crucial for the efficient likelihood routines
        union_vector = np.sort(np.array(tuple(union_set)))

        # if the union_vector is not shared by all experiments, the values of the
        # likelihood model's sensors in each experiment will be interpolated at the
        # values of the union vector; additionally, in each experiment the values of
        # the given correlation_variable will be overwritten with the union_vector
        if not shared_by_all:
            logger.warning(
                f"Experimental data must be interpolated in likelihood model "
                f"'{self.name}' since the '{correlation_variable}' vector differs "
                f"across the likelihood model's experiments."
            )
            for exp in self.experiment_names:
                for lm_sensor_name in self.sensor_names:
                    sensor_name = self.problem_experiments[exp]["correlation_info"][
                        lm_sensor_name
                    ][correlation_variable]
                    x = union_vector
                    xp = self.problem_experiments[exp]["sensor_values"][sensor_name]
                    yp = self.problem_experiments[exp]["sensor_values"][lm_sensor_name]
                    y = np.interp(x, xp, yp)
                    self.problem_experiments[exp]["sensor_values"][lm_sensor_name] = y
                for lm_sensor_name in self.sensor_names:
                    sensor_name = self.problem_experiments[exp]["correlation_info"][
                        lm_sensor_name
                    ][correlation_variable]
                    self.problem_experiments[exp]["sensor_values"][
                        sensor_name
                    ] = union_vector

        return union_vector

    def spatial_coordinate_array(self, spatial_variables: str) -> np.ndarray:
        """
        Returns a coordinate array for specified spatial correlation variables of the
        likelihood model. Note that these variables must appear in each of the
        likelihood model's experiments.

        Parameters
        ----------
        spatial_variables
            A string containing single appearances of the standard spatial coordinate
            characters 'x', 'y' and 'z'. For example spatial_variables = 'xz'.

        Returns
        -------
        coords
            An (n, d)-array where n is the number of spatial points and d the number of
            dimensions, i.e., the number of characters in spatial_variables.
        """

        # check that the spatial variables are the same across experiments
        for lm_sensor_name in self.sensor_names:
            for spatial_variable in spatial_variables:
                first_exp_name = self.experiment_names[0]
                first_experiment = self.problem_experiments[first_exp_name]
                s_values_1 = first_experiment["sensor_values"]
                corr_info_1 = first_experiment["correlation_info"]
                local_name_1 = corr_info_1[lm_sensor_name][spatial_variable]
                values_1 = s_values_1[local_name_1]
                for exp_name in self.experiment_names[1:]:
                    s_values = self.problem_experiments[exp_name]["sensor_values"]
                    corr_info = self.problem_experiments[exp_name]["correlation_info"]
                    local_name = corr_info[lm_sensor_name][spatial_variable]
                    values = s_values[local_name]
                    same_lengths = len_or_one(values) == len_or_one(values_1)
                    if (not same_lengths) or (not np.allclose(values, values_1)):
                        raise RuntimeError(
                            f"The sensor_values of the spatial correlation variable "
                            f"'{spatial_variable}' associated with the likelihood "
                            f"model's ('{self.name}') sensor '{lm_sensor_name}' differ "
                            f"across the likelihood model's experiments!"
                        )

        # after the previous check it suffices to focus on the first experiment
        first_exp_name = self.experiment_names[0]
        sensor_values = self.problem_experiments[first_exp_name]["sensor_values"]
        corr_info_1 = self.problem_experiments[first_exp_name]["correlation_info"]

        # assemble the spatial coordinate array
        arrays = []
        for lm_sensor_name in self.sensor_names:
            local_name_1 = corr_info_1[lm_sensor_name][spatial_variables[0]]
            values_1 = sensor_values[local_name_1]
            coords = np.zeros((len_or_one(values_1), len(spatial_variables)))
            coords[:, 0] = values_1
            for i, spatial_variable in enumerate(spatial_variables[1:], start=1):
                local_name = corr_info_1[lm_sensor_name][spatial_variable]
                values_i = sensor_values[local_name]
                coords[:, i] = values_i
            arrays.append(coords)
        coords = np.concatenate(arrays, axis=0)
        return coords


class SpaceOrTimeCorrelatedModelError(CorrelatedModelError):
    """
    This class serves as a parent class for the two scipy-based likelihood models that
    account for correlation in either space or time (not both). These classes are:
    CorrelatedModelError1D and SpaceCorrelatedModelError2D3D.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

        # this attribute states the number of elements in the likelihood model's
        # response vector (i.e., the vector of the forward model response vector
        # averaged over the likelihood model's experiments); it will be set after
        # possibly conducting some interpolation operations on the experimental data
        self.n_averaged_response_vector = 0

        # this type of likelihood model has only one sensor
        if self.n_sensors > 1:
            raise RuntimeError(
                f"Likelihood models of type 'SpaceOrTimeCorrelatedModelError' can only "
                f"have one sensor. However, found {self.n_sensors} instead."
            )
        self.sensor = self.sensors[0]

    def averaged_residuals_vector(self, model_response_dict: dict) -> np.ndarray:
        """
        Computes the averaged residuals vector. Since in correlation problems it is
        generally averaged over experiments, the residuals_vector has to be averaged
        too, and not stacked.

        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.

        Returns
        -------
        residuals_vector
            A one-dimensional vector containing the averaged model residuals. The length
            of this vector is given by self.n_averaged_response_vector.
        """

        # allocate the residuals vector
        if self.n_averaged_response_vector == 0:
            n = len(model_response_dict[self.experiment_names[0]][self.sensor.name])
            self.n_averaged_response_vector = n
        residuals_vector = np.zeros(self.n_averaged_response_vector)

        # average the response over the likelihood model's experiments
        for exp_name in self.experiment_names:
            ye_dict = self.problem_experiments[exp_name]["sensor_values"]
            ym_dict = model_response_dict[exp_name]
            residuals = ym_dict[self.sensor.name] - ye_dict[self.sensor.name]
            residuals_vector += residuals
        residuals_vector /= self.n_experiments

        return residuals_vector

    def averaged_responses_and_residuals_vector(
        self, model_response_dict: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Similar to self.averaged_residuals_vector but this method additionally returns
        the averaged vector of model responses.

        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.

        Returns
        -------
        residuals_vector
            A one-dimensional vector containing the averaged model responses. The length
            of this vector is given by self.n_averaged_response_vector.
        residuals_vector
            A one-dimensional vector containing the averaged model residuals. The length
            of this vector is given by self.n_averaged_response_vector.
        """

        # allocate the residuals vector
        if self.n_averaged_response_vector == 0:
            n = len(model_response_dict[self.experiment_names[0]][self.sensor.name])
            self.n_averaged_response_vector = n
        responses_vector = np.zeros(self.n_averaged_response_vector)
        residuals_vector = np.zeros(self.n_averaged_response_vector)

        # average the response over the likelihood model's experiments
        for exp_name in self.experiment_names:
            ye_dict = self.problem_experiments[exp_name]["sensor_values"]
            ym_dict = model_response_dict[exp_name]
            responses_vector += ym_dict[self.sensor.name]
            residuals = ym_dict[self.sensor.name] - ye_dict[self.sensor.name]
            residuals_vector += residuals
        responses_vector /= self.n_experiments
        residuals_vector /= self.n_experiments

        return responses_vector, residuals_vector


class SpaceAndTimeCorrelatedModelError(CorrelatedModelError):
    """
    This class serves as a parent class for the two scipy-based likelihood models that
    account for correlation in both space and time (not both). These classes are:
    SpaceTimeCorrelatedModelError1D and SpaceTimeCorrelatedModelError2D3D.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

        # these vectors will be derived from the experimental data
        self.time_vector = np.array([])
        self.space_vector = np.array([])

        # the model response is always a vector; this attribute states if this vector
        # is to be understood as a time signal (True) or a space signal (False); it
        # will be derived in the __init__-routines of the subclasses
        self.model_response_over_time = True

    @property
    def space_correlation_variables(self) -> str:
        """Returns the spatial correlation variables in this likelihood model."""
        return self.correlation_variables.replace("t", "")

    def scalar_based_correlation_vector(self, correlation_variable: str) -> np.ndarray:
        """
        Returns the vector of the scalar values of a given correlation variable.

        Parameters
        ----------
        correlation_variable
            A single character stating a default correlation variable, i.e., 'x', 'y',
            'z' or 't'.

        Returns
        -------
            The vector of the requested scalar correlation variables.
        """
        vector = []
        for lm_sensor_name in self.sensor_names:
            scalars = set()
            for exp_name in self.experiment_names:
                sensor_name = self.problem_experiments[exp_name]["correlation_info"][
                    lm_sensor_name
                ][correlation_variable]
                value = self.problem_experiments[exp_name]["sensor_values"][sensor_name]
                if len_or_one(value) > 1:
                    raise ValueError(
                        f"The correlation variable '{sensor_name}' in experiment "
                        f"'{exp_name}' is not scalar! It contains {len(value)}"
                        f"elements."
                    )
                scalars.add(value)
            if len(scalars) > 1:
                raise RuntimeError(
                    f"The scalar value of correlation variable '{correlation_variable}'"
                    f" associated with the likelihood model's sensor '{lm_sensor_name}'"
                    f" is not unique across experiments!"
                )
            vector.append(list(scalars)[0])
        return np.array(vector)

    def residual_array(self, model_response_dict: dict) -> np.ndarray:
        """
        Computes the residuals (model response minus the corresponding experimental
        data) for all of the likelihood model's sensors over all of the likelihood
        model's experiments and returns them in an array format. Note that this array
        reflects the average over all experiments as in all correlated models here.

        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.

        Returns
        -------
        residuals_array
            An 2D array (n_space, n_time) containing the model response minus the
            corresponding experimental data.
        """

        # allocate the array
        n_rows = len(self.space_vector)
        n_cols = len(self.time_vector)
        residuals_array = np.zeros((n_rows, n_cols))

        # fill the array; note that the two blocks distinguished below only differ in
        # writing the residuals row-wise or column-wise
        i = 0
        if self.model_response_over_time:
            for sensor_name in self.sensor_names:
                for exp_name in self.experiment_names:
                    sensor_values = self.problem_experiments[exp_name]["sensor_values"]
                    residuals_array[i, :] += model_response_dict[exp_name][sensor_name]
                    residuals_array[i, :] -= sensor_values[sensor_name]
                residuals_array[i, :] /= self.n_experiments
                i += 1
        else:
            for sensor_name in self.sensor_names:
                for exp_name in self.experiment_names:
                    sensor_values = self.problem_experiments[exp_name]["sensor_values"]
                    residuals_array[:, i] += model_response_dict[exp_name][sensor_name]
                    residuals_array[:, i] -= sensor_values[sensor_name]
                residuals_array[:, i] /= self.n_experiments
                i += 1

        return residuals_array

    def response_and_residual_array(
        self, model_response_dict: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Similar to self.residual_array but this method additionally returns the array
        of the model response averaged over the likelihood model's experiments.

        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.

        Returns
        -------
        residuals_array
            An 2D array (n_space, n_time) containing the model response minus the
            corresponding experimental data.
        response_array
            An 2D array (n_space, n_time) containing the model response averaged over
            the likelihood model's experiments.
        """

        # allocate the array
        n_rows = len(self.space_vector)
        n_cols = len(self.time_vector)
        response_array = np.zeros((n_rows, n_cols))
        residuals_array = np.zeros((n_rows, n_cols))

        # fill the array; note that the two blocks distinguished below only differ in
        # writing the residuals row-wise or column-wise
        i = 0
        if self.model_response_over_time:
            for sensor_name in self.sensor_names:
                for exp_name in self.experiment_names:
                    sensor_values = self.problem_experiments[exp_name]["sensor_values"]
                    response_array[i, :] += model_response_dict[exp_name][sensor_name]
                    residuals_array[i, :] += model_response_dict[exp_name][sensor_name]
                    residuals_array[i, :] -= sensor_values[sensor_name]
                response_array[i, :] /= self.n_experiments
                residuals_array[i, :] /= self.n_experiments
                i += 1
        else:
            for sensor_name in self.sensor_names:
                for exp_name in self.experiment_names:
                    sensor_values = self.problem_experiments[exp_name]["sensor_values"]
                    response_array[:, i] += model_response_dict[exp_name][sensor_name]
                    residuals_array[:, i] += model_response_dict[exp_name][sensor_name]
                    residuals_array[:, i] -= sensor_values[sensor_name]
                response_array[:, i] /= self.n_experiments
                residuals_array[:, i] /= self.n_experiments
                i += 1

        return response_array, residuals_array


class CorrelatedModelError1D(SpaceOrTimeCorrelatedModelError):
    """
    This class serves as a parent class for the two scipy-based likelihood models that
    account for correlation effects either in time or in one spatial coordinate. These
    model-classes are AdditiveCorrelatedModelError1D and
    MultiplicativeCorrelatedModelError1D.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

        # make sure there is only one correlation variable defined (remember, this
        # likelihood model is for 1D correlation only)
        if len(self.correlation_variables) > 1:
            raise RuntimeError(
                f"For likelihood model 'AdditiveCorrelatedModelError1D' only one "
                f"correlation variable can be defined! Found self.correlation_variables"
                f" = {self.correlation_variables}."
            )
        self.correlation_variable = self.correlation_variables

        # get the values of the only correlation variable; note that these values don't
        # change because they are entirely derived from the experimental data
        self.corr_vector = self.unionize_correlation_variable(self.correlation_variable)


class SpaceCorrelatedModelError2D3D(SpaceOrTimeCorrelatedModelError):
    """
    This class serves as a parent class for the two scipy-based likelihood models that
    account for correlation effects in two or three spatial coordinates (and not in
    time). These model-classes are AdditiveSpaceCorrelatedModelError2D3D and
    MultiplicativeSpaceCorrelatedModelError2D3D.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

        # the spatial coordinate array will be needed for the correlation matrix
        self.space_vector = self.spatial_coordinate_array(self.correlation_variables)


class SpaceTimeCorrelatedModelError1D(SpaceAndTimeCorrelatedModelError):
    """
    This class serves as a parent class for the two scipy-based likelihood models that
    account for correlation effects both in time and in one spatial coordinate. These
    model-classes are AdditiveSpaceTimeCorrelatedModelError1D and
    MultiplicativeSpaceTimeCorrelatedModelError1D.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

        # set the time and space vector depending on how they are defined
        if self.time_given_as_vector:
            self.model_response_over_time = True
            self.time_vector = self.unionize_correlation_variable("t")
            self.space_vector = self.scalar_based_correlation_vector(
                self.space_correlation_variables
            )
        else:
            self.model_response_over_time = False
            self.time_vector = self.scalar_based_correlation_vector("t")
            self.space_vector = self.unionize_correlation_variable(
                self.space_correlation_variables
            )


class SpaceTimeCorrelatedModelError2D3D(SpaceAndTimeCorrelatedModelError):
    """
    This class serves as a parent class for the two scipy-based likelihood models that
    account for correlation effects both in time and in 2 or 3 spatial coordinates.
    These model-classes are AdditiveSpaceTimeCorrelatedModelError2D3D and
    MultiplicativeSpaceTimeCorrelatedModelError2D3D.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

        # set the time vector depending on how it is defined
        if self.time_given_as_vector:
            self.model_response_over_time = True
            self.time_vector = self.unionize_correlation_variable("t")
        else:
            self.model_response_over_time = False
            self.time_vector = self.scalar_based_correlation_vector("t")
        self.space_vector = self.spatial_coordinate_array(
            self.space_correlation_variables
        )


# ==================================================================================== #
#                              Additive likelihood models                              #
# ==================================================================================== #


class AdditiveUncorrelatedModelError(UncorrelatedModelError):
    """
    This is a likelihood model based on a multivariate normal distribution without any
    correlations, i.e., with a diagonal covariance matrix. Both the model error as well
    as the measurement error (if considered) are assumed to be additive.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
        """
        # compute the residuals over all experiments
        res_vector = self.stacked_residuals_vector(model_response_dict)
        # process the standard deviation(s)
        if prms["std_model"] <= 0:
            return worst_value
        if self.additive_measurement_error:
            if prms["std_measurement"] <= 0:
                return worst_value
            # this is the formula for the variance of the sum of two normal dist.
            var = prms["std_model"] ** 2 + prms["std_measurement"] ** 2
        else:
            var = prms["std_model"] ** 2
        # evaluate the Gaussian log-PDF with zero mean and a variance of 1/prec for
        # each error term and sum them up
        ll = -len(res_vector) / 2 * np.log(2 * np.pi * var)
        ll -= 0.5 / var * np.sum(np.square(res_vector))
        return ll


class AdditiveCorrelatedModelError1D(CorrelatedModelError1D):
    """
    This is a likelihood model based on a multivariate normal distribution with that
    includes correlation effects in time or in one spatial coordinate. Both the model
    error as well as the measurement error (if considered) are assumed to be additive.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
        """

        # compute the model residuals via a method from the parent class
        res_vector = self.averaged_residuals_vector(model_response_dict)

        # parameters for the model prediction error
        std_model = prms["std_model"]
        l_corr = prms["l_corr"]
        if (std_model <= 0) or (l_corr <= 0):
            return worst_value

        # parameter for the measurement error
        if self.additive_measurement_error:
            std_meas = prms["std_measurement"]
            if std_meas <= 0:
                return worst_value
        else:
            std_meas = None

        # efficient log-likelihood evaluation via tripy
        ll = chol_loglike_1D(res_vector, self.corr_vector, l_corr, std_model, std_meas)
        return ll


class AdditiveSpaceCorrelatedModelError2D3D(SpaceCorrelatedModelError2D3D):
    """
    This is a likelihood model based on a multivariate normal distribution with that
    includes correlation effects in more than one spatial coordinate. Time correlation
    effects are not considered. Both the model error as well as the measurement error
    (if considered) are assumed to be additive.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
        """

        # compute the model residuals via a method from the parent class
        res_vector = self.averaged_residuals_vector(model_response_dict)
        n = self.n_averaged_response_vector  # for abbreviation

        # parameters for the model prediction error
        std_model = prms["std_model"]
        l_corr = prms["l_corr"]
        if (std_model <= 0) or (l_corr <= 0):
            return worst_value

        # parameter for the measurement error
        std_meas = 0
        if self.additive_measurement_error:
            std_meas = prms["std_measurement"]
            if std_meas <= 0:
                return worst_value

        # assemble covariance matrix
        f_corr = lambda a: correlation_function(d=a, correlation_length=l_corr)
        cov_matrix = std_model ** 2 * correlation_matrix(self.space_vector, f_corr)
        if self.additive_measurement_error:
            cov_matrix += std_meas ** 2 * np.eye(n)

        # evaluate log-likelihood (no efficient algorithm available in this case)
        return _loglike_multivariate_normal(res_vector, cov_matrix)


class AdditiveSpaceTimeCorrelatedModelError1D(SpaceTimeCorrelatedModelError1D):
    """
    This is a likelihood model based on a multivariate normal distribution with that
    includes correlation effects in time and one spatial coordinate. Both the model
    error as well as the measurement error (if considered) are assumed to be additive.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
        """

        # compute the model residuals via a method from the parent class
        res_array = self.residual_array(model_response_dict)

        # parameters for the model prediction error
        std_model = prms["std_model"]
        l_corr_x = prms["l_corr_space"]
        l_corr_t = prms["l_corr_time"]
        if (std_model <= 0) or (l_corr_x <= 0) or (l_corr_t <= 0):
            return worst_value

        # parameter for the measurement error
        if self.additive_measurement_error:
            std_meas = prms["std_measurement"]
            if std_meas <= 0:
                return worst_value
        else:
            std_meas = None

        # efficient log-likelihood evaluation via tripy
        ll = kron_loglike_2D_tridiag(
            res_array,
            self.space_vector,
            self.time_vector,
            l_corr_x,
            std_model,
            l_corr_t,
            1,
            std_meas,
        )
        return ll


class AdditiveSpaceTimeCorrelatedModelError2D3D(SpaceTimeCorrelatedModelError2D3D):
    """
    This is a likelihood model based on a multivariate normal distribution with that
    includes correlation effects in time and 2 or 3 spatial coordinates. Both the model
    error as well as the measurement error (if considered) are assumed to be additive.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
        """

        # compute the model residuals via a method from the parent class
        res_array = self.residual_array(model_response_dict)

        # parameters for the model prediction error
        std_model = prms["std_model"]
        l_corr_x = prms["l_corr_space"]
        l_corr_t = prms["l_corr_time"]
        if (std_model <= 0) or (l_corr_x <= 0) or (l_corr_t <= 0):
            return worst_value

        # parameter for the measurement error
        if self.additive_measurement_error:
            std_meas = prms["std_measurement"]
            if std_meas <= 0:
                return worst_value
        else:
            # consistent with tripy interface
            std_meas = None

        # assemble the dense spatial covariance matrix
        f = lambda a: correlation_function(d=a, correlation_length=l_corr_x)
        spatial_cov_matrix = std_model ** 2 * correlation_matrix(self.space_vector, f)

        # get the main diagonal and off-diagonal of the time covariance matrix inverse
        d0_t, d1_t = inv_cov_vec_1D(self.time_vector, l_corr_t, 1.0)

        # efficient log-likelihood evaluation via tripy
        ll = kron_loglike_2D(res_array, spatial_cov_matrix, [d0_t, d1_t], std_meas)

        return ll


# ==================================================================================== #
#                           Multiplicative likelihood models                           #
# ==================================================================================== #


class MultiplicativeUncorrelatedModelError(UncorrelatedModelError):
    """
    This is a likelihood model based on a multivariate normal distribution without any
    correlations, i.e., with a diagonal covariance matrix. The model error is assumed to
    be multiplicative while the measurement error (if considered) is assumed additive.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
        """
        # compute the model response and the residuals
        y_model, residuals = self.stacked_responses_and_residuals_vector(
            model_response_dict
        )

        # process the standard deviation for the model error
        std_model = prms["std_model"]
        if std_model <= 0:
            return worst_value

        # compute the covariance matrix depending on the measurement error; note that
        # without correlation (as it is the case here) the covariance matrix is diagonal
        # and can hence be represented by a single vector; this is the case in the
        # computations below, i.e., cov_mtx is a vector
        cov_mtx = np.power(y_model * std_model, 2)
        if self.additive_measurement_error:
            std_meas = prms["std_measurement"]
            if std_meas <= 0:
                return worst_value
            cov_mtx += np.power(std_meas, 2)

        # finally, evaluate the log-likelihood
        n = self.n_stacked_response_vector
        inv_cov_mtx = 1 / cov_mtx
        log_det_cov_mtx = np.sum(np.log(cov_mtx))
        ll = -0.5 * (n * np.log(2.0 * np.pi) + log_det_cov_mtx)
        ll += -0.5 * np.sum(np.power(residuals, 2) * inv_cov_mtx)
        return ll


class MultiplicativeCorrelatedModelError1D(CorrelatedModelError1D):
    """
    This is a likelihood model based on a multivariate normal distribution with that
    includes correlation effects in time or in one spatial coordinate. The model error
    is assumed to be multiplicative while the measurement error (if considered) is
    assumed to be additive.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
        """

        # compute the averaged model responses and residuals
        y_model, residuals = self.averaged_responses_and_residuals_vector(
            model_response_dict
        )

        # parameters for the model prediction error
        std_model = prms["std_model"]
        l_corr = prms["l_corr"]
        if (std_model <= 0) or (l_corr <= 0):
            return worst_value

        # parameter for the measurement error
        if self.additive_measurement_error:
            std_meas = prms["std_measurement"]
            if std_meas <= 0:
                return worst_value
        else:
            std_meas = None

        # efficient log-likelihood evaluation via tripy
        ll = chol_loglike_1D(
            residuals, self.corr_vector, l_corr, std_model, std_meas, y_model=y_model
        )
        return ll


class MultiplicativeSpaceCorrelatedModelError2D3D(SpaceCorrelatedModelError2D3D):
    """
    This is a likelihood model based on a multivariate normal distribution with that
    includes correlation effects in more than one spatial coordinate. Time correlation
    effects are not considered. The model error is assumed to be multiplicative while
    the measurement error (if considered) is assumed to be additive.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
        """

        # compute the averaged model responses and residuals
        y_model, residuals = self.averaged_responses_and_residuals_vector(
            model_response_dict
        )

        # parameters for the model prediction error
        std_model = prms["std_model"]
        l_corr = prms["l_corr"]
        if (std_model <= 0) or (l_corr <= 0):
            return worst_value

        # parameter for the measurement error
        if self.additive_measurement_error:
            std_meas = prms["std_measurement"]
            if std_meas <= 0:
                return worst_value
        else:
            std_meas = None

        # assemble covariance matrix
        n = self.n_averaged_response_vector
        f_corr = lambda a: correlation_function(d=a, correlation_length=l_corr)
        cov_matrix = std_model ** 2 * correlation_matrix(self.space_vector, f_corr)
        cov_matrix = np.multiply(
            y_model.reshape(-1, 1), np.multiply(y_model, cov_matrix)
        )
        if self.additive_measurement_error:
            cov_matrix += std_meas ** 2 * np.eye(n)

        # evaluate log-likelihood (no efficient algorithm available in this case)
        return _loglike_multivariate_normal(residuals, cov_matrix)


class MultiplicativeSpaceTimeCorrelatedModelError1D(SpaceTimeCorrelatedModelError1D):
    """
    This is a likelihood model based on a multivariate normal distribution with that
    includes correlation effects in time and one spatial coordinate. The model error is
    multiplicative, while the measurement error (if considered) is additive.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
        """

        # compute the model response/residuals via a method from the parent class
        y_model, residuals = self.response_and_residual_array(model_response_dict)

        # parameters for the model prediction error
        std_model = prms["std_model"]
        l_corr_x = prms["l_corr_space"]
        l_corr_t = prms["l_corr_time"]
        if (std_model <= 0) or (l_corr_x <= 0) or (l_corr_t <= 0):
            return worst_value

        # parameter for the measurement error
        if self.additive_measurement_error:
            std_meas = prms["std_measurement"]
            if std_meas <= 0:
                return worst_value
        else:
            # in case of zero-residuals, a value of std_meas = 0 leads to a covariance
            # matrix that is not invertible; however, there might be a better option
            # how to handle this case compared to this solution
            std_meas = 1e-9

        # get the main diagonal and off-diagonal of the space covariance matrix inverse
        d0_x, d1_x = inv_cov_vec_1D(self.space_vector, l_corr_x, std_model)

        # get the main diagonal and off-diagonal of the time covariance matrix inverse
        d0_t, d1_t = inv_cov_vec_1D(self.time_vector, l_corr_t, 1.0)

        # efficient log-likelihood evaluation via tripy
        ll = chol_loglike_2D(residuals, [d0_x, d1_x], [d0_t, d1_t], std_meas, y_model)
        return ll


class MultiplicativeSpaceTimeCorrelatedModelError2D3D(
    SpaceTimeCorrelatedModelError2D3D
):
    """
    This is a likelihood model based on a multivariate normal distribution with that
    includes correlation effects in time and 2 or 3 spatial coordinates. The model error
    is multiplicative, while the measurement error (if considered) is additive.

    Parameters
    ----------
    prms_def
        Parameter names defining which parameters are used by the likelihood model. For
        example prms_def = ['mu', 'sigma']. To check out the other possible formats, see
        the explanation for the same parameter in probeye/definition/forward_model.py:
        ForwardModelBase.__init__.
    sensors
        These are the sensor objects which serve as output sensors in one of the
        problem's forward models, that the likelihood model should refer to. This means,
        the likelihood model should describe the model error between the model response
        for all sensors specified in this 'sensors'-argument and the corresponding
        experimental data.
    experiment_names
        The names of the experiments in the scope of the likelihood model.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables. This argument can be any combination of the
        characters 'x', 'y', 'z', 't', each one appearing at most once. Examples are:
        'x', 't', 'xy', 'yzt'.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str]],
        problem_experiments: dict,
        additive_measurement_error: bool,
        correlation_variables: str,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
        """

        # compute the model residuals via a method from the parent class
        y_model, residuals = self.response_and_residual_array(model_response_dict)

        # parameters for the model prediction error
        std_model = prms["std_model"]
        l_corr_x = prms["l_corr_space"]
        l_corr_t = prms["l_corr_time"]
        if (std_model <= 0) or (l_corr_x <= 0) or (l_corr_t <= 0):
            return worst_value

        # parameter for the measurement error
        if self.additive_measurement_error:
            std_meas = prms["std_measurement"]
            if std_meas <= 0:
                return worst_value
        else:
            # in case of zero-residuals, a value of std_meas = 0 leads to a covariance
            # matrix that is not invertible; however, there might be a better option
            # how to handle this case compared to this solution
            std_meas = 1e-9

        # assemble the dense spatial covariance matrix
        f = lambda a: correlation_function(d=a, correlation_length=l_corr_x)
        spatial_cov_matrix = std_model ** 2 * correlation_matrix(self.space_vector, f)

        # get the main diagonal and off-diagonal of the time covariance matrix inverse
        d0_t, d1_t = inv_cov_vec_1D(self.time_vector, l_corr_t, 1.0)

        # efficient log-likelihood evaluation via tripy
        ll = chol_loglike_2D(
            residuals, spatial_cov_matrix, [d0_t, d1_t], std_meas, y_model
        )
        return ll


def translate_likelihood_model(lm_def: GaussianLikelihoodModel) -> ScipyLikelihoodBase:
    """
    Translates a given instance of GaussianLikelihoodModel (which is essentially just a
    description of the likelihood model without any computing-methods) to a specific
    likelihood model object which does contain SciPy-based computing-methods.

    Parameters
    ----------
    lm_def
        An instance of GaussianLikelihoodModel which contains general information on the
        likelihood model but no computing-methods.
    Returns
    -------
    likelihood_computer
        An instance of a specific likelihood model class with computing capabilities.
        Examples for such classes are given above in this file.
    """

    # likelihood model selection based on the flags given in the likelihood definition
    if lm_def.additive_model_error:
        if not lm_def.considers_correlation:
            l_class = "Add_Uncorrelated"
        else:
            if lm_def.considers_only_space_correlation:
                if len(lm_def.correlation_variables) == 1:
                    l_class = "Add_Correlated_1D"
                else:
                    l_class = "Add_SpaceCorrelated_2D3D"
            elif lm_def.considers_only_time_correlation:
                l_class = "Add_Correlated_1D"
            else:
                if len(lm_def.correlation_variables) == 2:
                    l_class = "Add_SpaceTimeCorrelated_1D"
                else:
                    l_class = "Add_SpaceTimeCorrelated_2D3D"
    else:
        if not lm_def.considers_correlation:
            l_class = "Mul_Uncorrelated"
        else:
            if lm_def.considers_only_space_correlation:
                if len(lm_def.correlation_variables) == 1:
                    l_class = "Mul_Correlated_1D"
                else:
                    l_class = "Mul_SpaceCorrelated_2D3D"
            elif lm_def.considers_only_time_correlation:
                l_class = "Mul_Correlated_1D"
            else:
                if len(lm_def.correlation_variables) == 2:
                    l_class = "Mul_SpaceTimeCorrelated_1D"
                else:
                    l_class = "Mul_SpaceTimeCorrelated_2D3D"

    # this dict allows to map an assigned string from the if-cases above to a specific
    # likelihood model defined in this file; the class is not assigned directly to the
    # variable l_class from the if-clauses above to avoid that l_class can have multiple
    # different types (which leads to issues during type-checking)
    class_dict = {
        "Add_Uncorrelated": AdditiveUncorrelatedModelError,
        "Add_Correlated_1D": AdditiveCorrelatedModelError1D,
        "Add_SpaceCorrelated_2D3D": AdditiveSpaceCorrelatedModelError2D3D,
        "Add_SpaceTimeCorrelated_1D": AdditiveSpaceTimeCorrelatedModelError1D,
        "Add_SpaceTimeCorrelated_2D3D": AdditiveSpaceTimeCorrelatedModelError2D3D,
        "Mul_Uncorrelated": MultiplicativeUncorrelatedModelError,
        "Mul_Correlated_1D": MultiplicativeCorrelatedModelError1D,
        "Mul_SpaceCorrelated_2D3D": MultiplicativeSpaceCorrelatedModelError2D3D,
        "Mul_SpaceTimeCorrelated_1D": MultiplicativeSpaceTimeCorrelatedModelError1D,
        "Mul_SpaceTimeCorrelated_2D3D": MultiplicativeSpaceTimeCorrelatedModelError2D3D,
    }

    # this is where the translation happens
    likelihood_computer = class_dict[l_class](
        lm_def.prms_def,
        lm_def.sensors,
        experiment_names=lm_def.experiment_names,
        problem_experiments=lm_def.problem_experiments,
        additive_measurement_error=lm_def.additive_measurement_error,
        correlation_variables=lm_def.correlation_variables,
        correlation_model=lm_def.correlation_model,
        name=lm_def.name,
    )

    return likelihood_computer
