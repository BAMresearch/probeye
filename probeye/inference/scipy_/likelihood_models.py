# standard library imports
from typing import Union, List, Optional, TYPE_CHECKING

# third party imports
import numpy as np
from tripy.loglikelihood import chol_loglike_1D

# local imports
from probeye.definition.likelihood_model import GaussianLikelihoodModel

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.sensor import Sensor


class AdditiveUncorrelatedModelError(GaussianLikelihoodModel):
    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str], None] = None,
        problem_experiments: Optional[dict] = None,
        additive_measurement_error: bool = False,
        correlation_variables: str = "",
        correlation_model: str = "exp",
        correlation_dict: Optional[dict] = None,
        name: Optional[str] = None,
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
            correlation_dict=correlation_dict,
            name=name,
        )

        # problem_experiments is an attribute not set by the parent-class
        self.problem_experiments = problem_experiments  # type: ignore

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log(likelihood) of this model.

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
        # compute the model error; note that this mode has exactly one sensor
        res_vector = self.residuals_vector(model_response_dict)
        # process the standard deviation(s)
        std = prms["std_model"]
        if std <= 0:
            return worst_value
        if self.additive_measurement_error:
            if prms["std_measurement"] <= 0:
                return worst_value
            # this is the formula for the std. dev. of the sum of two normal dist.
            std = np.sqrt(prms["std_model"] ** 2 + prms["std_measurement"] ** 2)
        # the precision 'prec' is defined as the inverse of the variance, hence
        # prec = 1 / sigma**2 where sigma denotes the standard deviation
        prec = 1.0 / std ** 2.0
        # evaluate the Gaussian log-PDF with zero mean and a variance of 1/prec for
        # each error term and sum them up
        ll = -len(res_vector) / 2 * np.log(2 * np.pi / prec)
        ll -= 0.5 * prec * np.sum(np.square(res_vector))
        return ll


class AdditiveCorrelatedModelError1D(GaussianLikelihoodModel):
    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str], None] = None,
        problem_experiments: Optional[dict] = None,
        additive_measurement_error: bool = False,
        correlation_variables: str = "",
        correlation_model: str = "exp",
        correlation_dict: Optional[dict] = None,
        name: Optional[str] = None,
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
            correlation_dict=correlation_dict,
            name=name,
        )

        # problem_experiments is an attribute not set by the parent-class
        self.problem_experiments = problem_experiments  # type: ignore

        # extract the values of the correlation variable
        self.coords = self.coordinate_vector(correlation_variables)

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log(likelihood) of this model.

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

        # compute the model residuals via a method from the parent class
        res_vector = self.residuals_vector(model_response_dict)
        ones = np.ones(len(res_vector))

        # parameters for the model prediction error
        std_model = prms["std_model"]
        l_corr = prms["l_corr"]
        if (std_model <= 0) or (l_corr <= 0):
            return worst_value
        std_model *= ones

        # parameter for the measurement error
        if self.additive_measurement_error:
            std_meas = prms["std_measurement"]
            if std_meas <= 0:
                return worst_value
            std_meas *= ones
        else:
            std_meas = None

        # efficient log-likelihood evaluation via tripy
        ll = chol_loglike_1D(self.coords, res_vector, l_corr, std_model, std_meas)
        return ll


def translate_likelihood_model(
    like_def: GaussianLikelihoodModel,
) -> Union[AdditiveUncorrelatedModelError, AdditiveCorrelatedModelError1D]:
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
            like_computer_class = AdditiveUncorrelatedModelError
        else:
            if like_def.considers_only_space_correlation:
                like_computer_class = AdditiveCorrelatedModelError1D  # type: ignore
            elif like_def.considers_only_time_correlation:
                like_computer_class = AdditiveCorrelatedModelError1D  # type: ignore
            else:
                raise NotImplementedError("Likelihood model not implemented yet!")
    else:
        raise NotImplementedError("Likelihood model not implemented yet!")

    # this is where the translation happens
    likelihood_computer = like_computer_class(
        like_def.prms_def,
        like_def.sensors,
        experiment_names=like_def.experiment_names,
        problem_experiments=like_def.problem_experiments,
        additive_measurement_error=like_def.additive_measurement_error,
        correlation_variables=like_def.correlation_variables,
        correlation_model=like_def.correlation_model,
        correlation_dict=like_def.correlation_dict,
        name=like_def.name,
    )

    return likelihood_computer
