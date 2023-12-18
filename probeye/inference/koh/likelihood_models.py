# standard library
from typing import Union, Tuple

# third party imports
import numpy as np

# local imports
from probeye.definition.likelihood_model import GaussianLikelihoodModel

from probeye.inference.scipy.likelihood_models import (
    ScipyLikelihoodBase,
    UncorrelatedModelError,
)

# ==================================================================================== #
#                              Additive likelihood models                              #
# ==================================================================================== #


class KOHUncorrelatedModelError(UncorrelatedModelError):
    """
    This is a likelihood model based on a multivariate normal distribution without any
    correlations, i.e., with a diagonal covariance matrix. Both the model error as well
    as the measurement error (if considered) are assumed to be additive.

    Parameters
    ----------
    uncorrelated_model_error
        The UncorrelatedModelError instance to derive from.
    """

    def __init__(self, uncorrelated_model_error: UncorrelatedModelError):
        super().__init__(uncorrelated_model_error)

    def loglike(
        self,
        response_vector: np.ndarray,
        residual_vector: np.ndarray,
        prms: dict,
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (SolverLikelihoodBase).
        """

        # TODO Implement new base forward model that includes bias compulsory
        return float(self.forward_model.bias.gp.log_marginal_likelihood())

class MomentMatchingModelError(UncorrelatedModelError):
    def __init__(self, uncorrelated_model_error: UncorrelatedModelError):
        super().__init__(uncorrelated_model_error)
        self.tolerance = uncorrelated_model_error.tolerance
        self.gamma = uncorrelated_model_error.gamma

    def loglike(
        self,
        response_vector: np.ndarray,
        residual_vector: np.ndarray,
        prms: dict,
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (SolverLikelihoodBase).
        """
        # in this case, 'variance' is a scalar
        ll = -1 / 2 * np.log(2 * np.pi * self.tolerance**2)
        ll -= 0.5 / self.tolerance**2 * np.sum(np.square(residual_vector)+np.square(response_vector[1]-self.gamma*np.abs(residual_vector)))

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
    prefix = "Add" if lm_def.additive_model_error else "Mul"
    if not lm_def.considers_correlation:
        l_class = f"{prefix}_KOH_Uncorrelated"
    else:
        raise NotImplementedError

    # this dict allows to map an assigned string from the if-cases above to a specific
    # likelihood model defined in this file; the class is not assigned directly to the
    # variable l_class from the if-clauses above to avoid that l_class can have multiple
    # different types (which leads to issues during type-checking)
    class_dict = {
        "Add_KOH_Uncorrelated": KOHUncorrelatedModelError,
        "Mul_KOH_Uncorrelated": MomentMatchingModelError
    }

    # this is where the translation happens
    likelihood_computer = class_dict[l_class](lm_def)

    return likelihood_computer
