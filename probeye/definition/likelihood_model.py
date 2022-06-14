# standard library
from typing import Optional

# local imports
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.correlation_model import CorrelationModel
from probeye.subroutines import len_or_one


class GaussianLikelihoodModel:
    """
    This class describes a Gaussian (i.e., normal) likelihood model in general terms.
    It contains information such as the likelihood model's latent parameters, its scope
    with respect to a given experiment, the sensors it considers, error model specifics
    as well as information on possible correlations to be considered.

    Parameters
    ----------
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    model_error
        Either 'additive' or 'multiplicative'. This argument defines whether an additive
        or a multiplicative model prediction error should be assumed.
    measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    """

    def __init__(
        self,
        experiment_name: str,
        model_error: str,
        measurement_error: Optional[str] = None,
        correlation: Optional[CorrelationModel] = None,
    ):

        self.prms_def = {}  # type: dict
        self.prms_dim = 0

        # a likelihood model refers to exactly one experiment
        self.experiment_name = experiment_name

        # the likelihood model's forward model will be derived and set by the method
        # InverseProblem.add_likelihood_model from the given experiment_name
        self.forward_model = ForwardModelBase(name="_dummy_")

        # attributes related to the error model
        if model_error == "additive":
            self.additive_model_error = True
            self.multiplicative_model_error = False
        elif model_error == "multiplicative":
            self.additive_model_error = False
            self.multiplicative_model_error = True
        else:
            raise ValueError(
                f"Found an invalid value for argument 'model_error' ('{model_error}') "
                f"in the likelihood model for experiment '{self.experiment_name}'. "
                f"Valid options are: 'additive', 'multiplicative'"
            )
        self.measurement_error = measurement_error

        # correlation-related attributes from the given input
        self.correlation_model = correlation

        # note that a correlation variable can have multiple dimensions; for example,
        # a likelihood model with the spatial correlation variables 'x' and 'y' has one
        # and not two correlation variables
        self.considers_correlation = True if (correlation is not None) else False
        self.n_correlation_variables = 0
        if correlation is not None:
            self.n_correlation_variables = len(correlation.correlation_variables)

        # check if there is a multidimensional correlation variable; currently, such
        # correlation variables are understood as spatial coordinates
        self.has_S23D_correlation_variable = False
        if correlation is not None:
            for corr_var in correlation.correlation_variables:
                if type(corr_var) in [list, tuple]:
                    if len(corr_var) > 1:
                        self.has_S23D_correlation_variable = True

        # this attribute will be set when calling self._determine_output_lengths, which
        # is possible as soon as the forward model was assigned to the likelihood model
        self.output_lengths = {}  # type: dict

    @property
    def correlation_variables(self) -> list:
        """Shortens the access of the correlation model's correlation variables."""
        if self.correlation_model is not None:
            return self.correlation_model.correlation_variables
        else:
            return []

    def determine_output_lengths(self):
        """
        Sets the self.output_lengths dictionary. This dict contains information on the
        length of the returned values of the likelihood model's forward model in the
        likelihood model's experiment. A simple example for an uncorrelated case could
        look like this (note that the ':'-character is the key for the full response):
        {':': {'total': 202, 'increments': [101, 101], 'names': ['y1', 'y2']}}
        This is interpreted as follows: for the likelihood's experiment, the forward
        model's output dictionary will eventually be translated into a vector holding
        202 values, where the first 101 belong to output sensor 'y1' and the following
        101 values belong to output sensor. In a correlated case, the created dict will
        additionally contain the lengths of the correlation variables, e.g.:
        {':': {'total': 12, 'increments': [6, 6], 'names': ['y1', 'y2']},
         't': {'total': 2,  'increments': [1, 1], 'names': ['y1', 'y2']},
         'x': {'total': 12, 'increments': [6, 6], 'names': ['y1', 'y2']}}
        The 't' and 'x' entries are interpreted as the 't'-correlation vector having
        length 2 and the 'x'-correlation vector having length 12, while the remaining
        information is interpreted analogously as described before.
        """

        # add the information for the full model response; the key of this full response
        # is ':' which was chosen because it is an unlikely name for a correlation var.
        output_lengths = {
            ":": {
                "total": 0,
                "increments": [],
                "names": [],
            }
        }
        for output_sensor in self.forward_model.output_sensors:
            n_i = len_or_one(output_sensor[self.experiment_name])
            output_lengths[":"]["increments"].append(n_i)
            output_lengths[":"]["names"].append(output_sensor.name)
        output_lengths[":"]["total"] = sum(output_lengths[":"]["increments"])

        # add the information for the correlation vectors
        if self.considers_correlation:
            for corr_var_ in self.correlation_variables:
                corr_var_tuple = corr_var_
                if isinstance(corr_var_, str):
                    corr_var_tuple = (corr_var_,)
                for corr_var in corr_var_tuple:
                    output_lengths[corr_var] = {
                        "total": 0,
                        "increments": [],
                        "names": [],
                    }
                    for os in self.forward_model.output_sensors:
                        if hasattr(os, corr_var) and getattr(os, corr_var) is not None:
                            n_i = len_or_one(getattr(os, corr_var))
                        else:
                            n_i = len_or_one(os[self.experiment_name])
                        output_lengths[corr_var]["increments"].append(n_i)
                        output_lengths[corr_var]["names"].append(os.name)
                    output_lengths[corr_var]["total"] = sum(
                        output_lengths[corr_var]["increments"]
                    )

        # write the information to its attribute
        self.output_lengths = output_lengths
