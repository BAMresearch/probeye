# standard library
from typing import Union, Tuple

# third party imports
import numpy as np
import math 
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
        self.bias_model = "koh"

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
    
class EmbeddedUncorrelatedModelError(UncorrelatedModelError):
    def __init__(self, uncorrelated_model_error: UncorrelatedModelError):
        super().__init__(uncorrelated_model_error)
        self.bias_model = "embedded"

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
        raise NotImplementedError

class MomentMatchingModelError(EmbeddedUncorrelatedModelError):
    def __init__(self, uncorrelated_model_error: UncorrelatedModelError):
        super().__init__(uncorrelated_model_error)
        self.tolerance = uncorrelated_model_error.tolerance
        self.gamma = uncorrelated_model_error.gamma
        self.l_model = "moment_matching"

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
        # Uses Sargsyan2019 Eq. 15 as ABC likelihood function
        # FIXME: modified to incorporate the noise model, now through the tolerance
        if not hasattr(self, "weight_mean"):
            self.weight_mean = 1
        if not hasattr(self, "weight_std"):
            self.weight_std = 1
        std_model, std_meas, stds_are_scalar = self.std_values(prms)
        variance = np.power(std_model, 2)
        n = len(residual_vector)
        ll = 0
        # ll -= 0.5 / self.tolerance**2 * np.sum(np.square(residual_vector)+np.square(response_vector[1]-self.gamma*np.abs(residual_vector)))
        if std_meas is not None:
            variance += np.power(std_meas, 2)
        if stds_are_scalar:
            # std_vector = np.full_like(residual_vector, np.sqrt(variance))
            # ll = -1 / 2 * np.log(2 * np.pi * self.tolerance**2)
            # ll -= 0.5 / self.tolerance**2 * np.sum(np.square(residual_vector)+np.square(np.sqrt(np.square(response_vector[1])+np.square(std_vector))-self.gamma*np.abs(residual_vector)))
            # ll -= -n / 2 * np.log(2 * np.pi * variance)
            # ll -= 0.5 / variance * np.sum(np.square(residual_vector))
            std_vector = np.full_like(residual_vector, np.sqrt(variance))
            ll = -0.5*n * np.log(2 * np.pi * self.tolerance**2 * variance)
            # # ll = -0.5 * np.log(2 * np.pi * self.tolerance**2)
            # ll = -0.5 * np.log(2 * np.pi * variance)
            # ll -= 0.5* np.sum(np.square(self.weight_mean*residual_vector)/variance+np.square(self.weight_std*np.sqrt(np.square(response_vector[1])+np.square(std_vector))-self.gamma*np.abs(residual_vector))/self.tolerance**2)
            ll -= 0.5/variance* np.sum(np.square(self.weight_mean*residual_vector)+np.square(self.weight_std*np.sqrt(np.square(response_vector[1])+np.square(std_vector))-self.gamma*np.abs(residual_vector)))
        else:
            ll -= -0.5 * (n * np.log(2 * np.pi) + np.sum(np.log(variance)))
            ll -= 0.5 * np.sum(np.square(residual_vector) / variance)

        if hasattr(self, "moment_residuals"):
            self.moment_residuals["mean"].append(np.sum(residual_vector))
            self.moment_residuals["std"].append(np.sum(np.sqrt(np.square(response_vector[1])+np.square(std_vector))-self.gamma*np.abs(residual_vector)))
        return ll
    
class GlobalMomentMatchingModelError(EmbeddedUncorrelatedModelError):
    def __init__(self, uncorrelated_model_error: UncorrelatedModelError):
        super().__init__(uncorrelated_model_error)
        self.gamma = uncorrelated_model_error.gamma
        self.l_model = "global_moment_matching"

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

        std_model, std_meas, stds_are_scalar = self.std_values(prms)
        variance = np.power(std_model, 2)
        n = len(residual_vector)
        mean_residual = np.mean(residual_vector)
        variance_residual = np.var(residual_vector)
        mean_var = np.mean(np.square(response_vector[1]))
        sample_variance = mean_var + variance_residual + variance
        population_variance = self.gamma**2*np.mean(np.square(residual_vector))+ variance_residual + variance
        ll = 0
        if std_meas is not None:
            variance += np.power(std_meas, 2)
        if stds_are_scalar:
            # ll -=0.5 * np.log(2 * np.pi / n * variance)
            ll -=0.5 * np.log(2 * np.pi / n * population_variance)
            ll -= 0.5 * n / population_variance * np.square(mean_residual)
            # ll -= 0.5 * n / variance * np.square(mean_residual)
            ll -= 0.5 * n * sample_variance / population_variance
            ll -= (n-1) / 2 * np.log(2) 
            ll -= math.lgamma((n-1)/2)
            ll += ((n-1)/2-1) * np.log(n*sample_variance/population_variance)
        return ll
    
class RelativeGlobalMomentMatchingModelError(EmbeddedUncorrelatedModelError):
    def __init__(self, uncorrelated_model_error: UncorrelatedModelError):
        super().__init__(uncorrelated_model_error)
        self.gamma = uncorrelated_model_error.gamma
        self.l_model = "relative_global_moment_matching"

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

        std_model, std_meas, stds_are_scalar = self.std_values(prms)
        variance = np.power(std_model, 2)
        n = len(residual_vector)
        sigma_model_population = np.sqrt(self.gamma**2 * np.square(residual_vector))
        sigma_model_sample = np.sqrt(np.square(response_vector[1])+variance)
        population_variance = (np.var(np.divide(residual_vector, sigma_model_population)) + 1)
                            #    std_model**2*np.var(np.reciprocal(sigma_model_population)) +
                            #    np.square(1+std_model*np.mean(np.reciprocal(sigma_model_population))))
        sample_variance = (np.var(np.divide(residual_vector, sigma_model_sample)) + 1)
                            #    std_model**2*np.var(np.reciprocal(sigma_model_sample)) +
                            #    np.square(1+std_model*np.mean(np.reciprocal(sigma_model_sample))))
        mean_residual = np.mean(np.divide(residual_vector, sigma_model_sample))

        ll = 0
        if std_meas is not None:
            variance += np.power(std_meas, 2)
        if stds_are_scalar:
            # ll -=0.5 * np.log(2 * np.pi / n * variance)
            ll -=0.5 * np.log(2 * np.pi / n * population_variance)
            ll -= 0.5 * n / population_variance * np.square(mean_residual)
            # ll -= 0.5 * n / variance * np.square(mean_residual)
            ll -= 0.5 * n * sample_variance / population_variance
            ll -= (n-1) / 2 * np.log(2) 
            ll -= math.lgamma((n-1)/2)
            ll += ((n-1)/2-1) * np.log(n*sample_variance/population_variance)
        return ll
    
class IndependentNormalModelError(EmbeddedUncorrelatedModelError):
    def __init__(self, uncorrelated_model_error: UncorrelatedModelError):
        super().__init__(uncorrelated_model_error)
        self.l_model = "independent_normal"

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

        std_model, std_meas, stds_are_scalar = self.std_values(prms)
        variance = np.power(std_model, 2)
        n = len(residual_vector)
        sigma_model_sample = np.sqrt(np.square(response_vector[1])+variance)

        ll = 0
        if std_meas is not None:
            variance += np.power(std_meas, 2)
        if stds_are_scalar:
            ll -=0.5 * n * np.log(2 * np.pi )
            ll -= np.sum(np.log(sigma_model_sample))
            ll -= np.sum(np.square(residual_vector)/(2*np.square(sigma_model_sample)))
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
    prefix = "KOH" if lm_def.bias_model == "koh" else "Embedded"
    if not lm_def.considers_correlation:
        l_class = f"{prefix}_{lm_def.l_model}_Uncorrelated"
    else:
        raise NotImplementedError

    # this dict allows to map an assigned string from the if-cases above to a specific
    # likelihood model defined in this file; the class is not assigned directly to the
    # variable l_class from the if-clauses above to avoid that l_class can have multiple
    # different types (which leads to issues during type-checking)
    class_dict = {
        "Add_KOH_Uncorrelated": KOHUncorrelatedModelError,
        "Embedded_moment_matching_Uncorrelated": MomentMatchingModelError,
        "Embedded_global_moment_matching_Uncorrelated": GlobalMomentMatchingModelError,
        "Embedded_relative_global_moment_matching_Uncorrelated": RelativeGlobalMomentMatchingModelError,
        "Embedded_independent_normal_Uncorrelated": IndependentNormalModelError
    }

    # this is where the translation happens
    likelihood_computer = class_dict[l_class](lm_def)

    return likelihood_computer
