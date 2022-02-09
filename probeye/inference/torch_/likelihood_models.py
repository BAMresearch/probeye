# standard library imports
from typing import Union, List, Optional, TYPE_CHECKING


# third party imports
import torch as th
import pyro
import pyro.distributions as dist

# local imports
from probeye.subroutines import len_or_one
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.sensor import Sensor


class AdditiveUncorrelatedModelError(GaussianLikelihoodModel):
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
        """
        For a detailed explanation of the input arguments check out the docstring given
        in probeye/definition/likelihood_models.py:GaussianLikelihoodModel. The only
        additional argument is 'problem_experiments' which is simply a pointer to
        InferenceProblem._experiments (a dictionary of all the problem's experiments).
        """

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

    def residuals_vector(self, model_response_dict: dict) -> th.Tensor:
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
        residuals_vector = th.zeros(n)
        idx = 0
        for residuals_sub_vector in residuals_dict.values():
            m = len_or_one(residuals_sub_vector)
            residuals_vector[idx : idx + m] = residuals_sub_vector
            idx += m
        return residuals_vector

    def sample_cond_likelihood(self, model_response: dict, prms: dict) -> pyro.sample:
        """
        Creates a likelihood-sample conditioned on the observed errors.
        model_response_dict
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
