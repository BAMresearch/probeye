# standard library
from typing import Union, List, Optional

# local imports
from probeye.definition.forward_model import ForwardModelBase
from probeye.subroutines import translate_prms_def, make_list


class GaussianLikelihoodModel:
    """
    This class describes a Gaussian (i.e., normal) likelihood model in general terms.
    It contains information such as the likelihood model's latent parameters, its scope
    with respect to a given experiment, the sensors it considers, error model specifics
    as well as information on possible correlations to be considered.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    model_error
        Either 'additive' or 'multiplicative'. This argument defines whether an additive
        or a multiplicative model prediction error should be assumed.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
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
        prms_def: Union[str, List[str], dict],
        experiment_name: str,
        model_error: str = "additive",
        additive_measurement_error: bool = False,
        correlation_variables: Optional[Union[str, list]] = None,
        correlation_model: str = "exp",
        name: str = "",
    ):

        # general attributes
        self.name = name
        self.prms_def, self.prms_dim = translate_prms_def(prms_def)
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
                f"Found invalid value for model_error '{model_error}' in likelihood "
                f"model '{self.name}'. Valid options are: 'additive', 'multiplicative'"
            )
        self.additive_measurement_error = additive_measurement_error

        # correlation-related attributes from the given input
        self.correlation_variables = []
        if correlation_variables is not None:
            self.correlation_variables = make_list(correlation_variables)
        self.correlation_model = correlation_model
        self.valid_correlation_models = ("exp",)

        # the following attributes are set in self.process_correlation_definition
        self.considers_correlation = False
        self.n_correlation_variables = 0
        self.has_S23D_correlation_variable = False

    def process_correlation_definition(self):
        """
        Processes and checks the correlation information for the likelihood model, which
        is given by the attributes of the likelihood model itself and by the information
        given in the sensors of the likelihood model's forward model.
        """

        # check if a valid correlation model was specified
        if self.correlation_model not in self.valid_correlation_models:
            raise ValueError(
                f"Found invalid correlation model '{self.correlation_model}' in the "
                f"correlation definition. Currently, valid correlation models "
                f"are: {self.valid_correlation_models}."
            )

        # note that a correlation variable can have multiple dimensions; for example,
        # a likelihood model with the spatial correlation variables 'x' and 'y' has one
        # and not two correlation variables
        self.considers_correlation = True if self.correlation_variables else False
        self.n_correlation_variables = len(self.correlation_variables)

        # check if there is a multidimensional correlation variable; currently, such
        # correlation variables are understood as spatial coordinates
        for corr_var in self.correlation_variables:
            if type(corr_var) in [list, tuple]:
                if len(corr_var) > 1:
                    self.has_S23D_correlation_variable = True
