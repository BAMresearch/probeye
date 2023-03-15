# standard library
from typing import Union, Tuple

# third party imports
import numpy as np

# local imports
from probeye.definition.likelihood_model import GaussianLikelihoodModel

from probeye.inference.scipy.likelihood_models import ScipyLikelihoodBase, UncorrelatedModelError

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
        # compute the variance vector (the diagonal of the covariance matrix) or scalar
        std_model, std_meas, stds_are_scalar = self.std_values(prms)
        variance = np.power(std_model, 2)
        n = len(residual_vector)
        if std_meas is not None:
            variance += np.power(std_meas, 2)
        if stds_are_scalar:
            # in this case, 'variance' is a scalar
            ll = -n / 2 * np.log(2 * np.pi * variance)
            ll -= 0.5 / variance * np.sum(np.square(residual_vector))
        else:
            # in this case, 'variance' is a  (non-constant) vector
            ll = -0.5 * (n * np.log(2 * np.pi) + np.sum(np.log(variance)))
            ll -= 0.5 * np.sum(np.square(residual_vector) / variance)

        #TODO Implement new base forward model that includes bias compulsory
        # return float(self.forward_model.bias.gp.log_marginal_likelihood()) - ll
        return float(self.forward_model.bias.gp.log_marginal_likelihood())


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
        # if lm_def.n_correlation_variables == 1:
        #     if lm_def.has_S23D_correlation_variable:
        #         l_class = f"{prefix}_Correlated_S23D"
        #     else:
        #         l_class = f"{prefix}_Correlated_1D"
        # else:
        #     if lm_def.has_S23D_correlation_variable:
        #         l_class = f"{prefix}_Correlated_1DS23D"
        #     else:
        #         l_class = f"{prefix}_Correlated_1D1D"
        raise NotImplementedError

    # this dict allows to map an assigned string from the if-cases above to a specific
    # likelihood model defined in this file; the class is not assigned directly to the
    # variable l_class from the if-clauses above to avoid that l_class can have multiple
    # different types (which leads to issues during type-checking)
    class_dict = {
        "Add_KOH_Uncorrelated": KOHUncorrelatedModelError,
    }

    # this is where the translation happens
    likelihood_computer = class_dict[l_class](lm_def)

    return likelihood_computer
