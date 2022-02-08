# standard library
from typing import Union, List, Optional

# local imports
from probeye.definition.sensor import Sensor
from probeye.definition.forward_model import ForwardModelBase
from probeye.subroutines import make_list
from probeye.subroutines import translate_prms_def


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
        name: str = "",
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
            The names of the experiments in the scope of the likelihood model.
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

        # correlation-related attributes from the given input
        self.correlation_variables = correlation_variables
        self.correlation_model = correlation_model

        # derived correlation variables
        self.n_correlation_variables = 0
        self.considers_correlation = False
        self.considers_time_correlation = False
        self.considers_space_correlation = False
        self.considers_only_time_correlation = False
        self.considers_only_space_correlation = False
        self.considers_space_and_time_correlation = False
        self.considers_time_and_space_correlation = False
        self.process_correlation_definition()

        # instantiate the experiment's names as a list
        self.experiment_names = []
        if experiment_names is not None:
            self.experiment_names = make_list(experiment_names)

        # these attributes are not directly set by the user, but derived and set by the
        # method InferenceProblem.add_likelihood_model
        self.forward_model = None  # type: Optional[ForwardModelBase]
        self.problem_experiments = {}  # type: dict

    @property
    def n_experiments(self) -> int:
        """
        Provides a dynamic attributes stating the number of experiments that
        were assigned to the log-likelihood model.
        """
        return len(self.experiment_names)

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

        # this check is mostly to prevent automatic type checking issues
        if self.problem_experiments is None:
            raise ValueError(
                f"The attribute likelihood_model.problem_experiments has not been set "
                f"yet. This should have happened automatically. Something general "
                f"seems to have gone wrong here..."
            )

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

        # this sets self.forward_model if not set yet
        self.determine_forward_model()

    def check_experiment_consistency(self):
        """
        Checks if the experiments defined on the likelihood model are consistent with
        each other. Mostly, this means that they all refer to the same forward model.
        """

        # obviously, there has to be at least one experiment defined
        if self.experiment_names is None:
            raise RuntimeError(
                f"No experiments defined for likelihood model '{self.name}'!"
            )

        # all experiments must refer to the same forward model
        fwd_models = set()
        for exp_name in self.experiment_names:
            fwd_models.add(self.problem_experiments[exp_name]["forward_model"])
        if len(fwd_models) > 1:
            raise RuntimeError(
                f"The experiments of likelihood model '{self.name}' refer to more than "
                f"one forward model!\nHowever, they should all refer to one and the "
                f"same forward model."
            )

    def determine_forward_model(self):
        """
        Determines the forward model of the likelihood model as the forward model that
        is referenced by all of its experiments. Note, that all experiments of a
        likelihood model must refer to one and only one common forward model.
        """
        # note that the check makes sure that all experiments have the same forward
        # model; hence we can simply take any experiment and read its forward model
        self.check_experiment_consistency()
        exp_1 = self.problem_experiments[self.experiment_names[0]]
        self.forward_model = exp_1["forward_model"]

    def process_correlation_definition(self, valid_corr_models: tuple = ("exp",)):
        """
        Processes a string like 'xt', 'xy', etc. into the corresponding correlation
        variables and decides if a mere spatial, temporal or spatio-temporal correlation
        was defined. The results are written to attributes. This function is intended
        to be called right after initializing the likelihood model without having added
        any experiments yet.

        Parameters
        ----------
        valid_corr_models
            The tuple contains all currently implemented correlation models.
        """

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

        # translate the given string in a list of its characters
        cv_list = list(self.correlation_variables)

        # check that only valid characters are given, and that those characters
        # are at most mentioned once
        for char in cv_list:
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
        self.n_correlation_variables = len(self.correlation_variables)

        # this is where the actual processing happens
        self.considers_correlation = True if self.correlation_variables else False
        self.considers_time_correlation = "t" in self.correlation_variables
        self.considers_space_correlation = (
            len({"x", "y", "z"}.intersection(set(self.correlation_variables))) > 0
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
