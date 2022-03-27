# standard library imports
from typing import Union, List, TYPE_CHECKING


# third party imports
import torch as th
import pyro
import pyro.distributions as dist

# local imports
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.sensor import Sensor


class AdditiveUncorrelatedModelError(GaussianLikelihoodModel):
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

        # initialize the super-class (GaussianLikelihoodModel) based on the given input
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

        # problem_experiments is an attribute not set by the parent-class
        self.problem_experiments = problem_experiments  # type: ignore

    def residuals(self, model_response_dict: dict) -> dict:
        """
        Computes the residuals for all of the likelihood model's experiments and returns
        them in a dictionary that is sorted by output sensor_values. This method
        overwrites the corresponding method of the parent class.

        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.

        Returns
        -------
        model_error
            A dictionary with the keys being the likelihood model's sensor names, and
            torch.Tensors representing the model errors as values.
        """

        # prepare the dictionary keys
        residuals_dict = {name: th.Tensor([]) for name in self.sensor_names}

        # fill the dictionary with model error vectors
        for exp_name in self.experiment_names:
            exp_dict = self.problem_experiments[exp_name]  # type: ignore
            ym_dict = model_response_dict[exp_name]
            ye_dict = exp_dict["sensor_values"]
            for sensor_name in self.sensor_names:
                residuals_vector = ym_dict[sensor_name] - ye_dict[sensor_name]
                residuals_dict[sensor_name] = th.cat(
                    (residuals_dict[sensor_name], residuals_vector)
                )
        return residuals_dict

    def sample_cond_likelihood(self, model_response: dict, prms: dict):
        """
        Creates a likelihood-sample conditioned on the observed errors.

        Parameters
        ----------
        model_response
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.
        prms
            Contains parameter name:value pairs.
        """
        std = prms["std_model"]
        # compute the model error; note that this mode has exactly one sensor
        res_vector = self.residuals(model_response)[self.sensors[0].name]
        pyro.sample(f"lkl_{self.name}", dist.Normal(0, std), obs=res_vector)


def translate_likelihood_model(
    like_def: GaussianLikelihoodModel,
) -> AdditiveUncorrelatedModelError:
    """
    Translates a given instance of GaussianLikelihoodModel (which is essentially just a
    description of the likelihood model without any computing-methods) to a specific
    likelihood model object which does contain SciPy-based computing-methods.

    Parameters
    ----------
    like_def
        An instance of GaussianLikelihoodModel which contains general information on the
        likelihood model but no computing-methods.

    Returns
    -------
    likelihood_computer
        An instance of a specific likelihood model class with computing capabilities.
        Examples for such classes are given above in this file.
    """

    # likelihood model selection based on the flags given in the likelihood definition
    if like_def.additive_model_error:
        if not like_def.considers_correlation:
            likelihood_computer_class = AdditiveUncorrelatedModelError
        else:
            raise NotImplementedError("Likelihood model not implemented yet!")
    else:
        raise NotImplementedError("Likelihood model not implemented yet!")

    # this is where the translation happens
    likelihood_computer = likelihood_computer_class(
        like_def.prms_def,
        like_def.sensors,
        experiment_names=like_def.experiment_names,
        problem_experiments=like_def.problem_experiments,
        additive_measurement_error=like_def.additive_measurement_error,
        correlation_variables=like_def.correlation_variables,
        correlation_model=like_def.correlation_model,
        name=like_def.name,
    )

    return likelihood_computer
