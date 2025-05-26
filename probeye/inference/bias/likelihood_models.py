# standard library
# third party imports
import numpy as np
import math

# local imports
from probeye.definition.likelihood_model import GaussianLikelihoodModel

from probeye.inference.scipy.likelihood_models import (
    ScipyLikelihoodBase,
    UncorrelatedModelError,
)


class EmbeddedLikelihoodBaseModel(ScipyLikelihoodBase):
    """
    This class implements the embedded likelihood model base class.

    Parameters
    ----------
    l_model
        The likelihood model used in this class. This is set to "moment_matching". Works as a flag to
        distinguish between the different likelihood models.
    tolerance
        The tolerance value used in the moment matching likelihood model.
    gamma
        The gamma value used in the moment matching likelihood model.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each

    Attributes
    ----------
    bias_model
        The bias model used in this class. This is set to "embedded". Works as a flag to
        distinguish between the different bias models.
    """

    def __init__(
        self,
        l_model: str = "moment_matching",
        tolerance: float = 1e-6,
        gamma: float = 1.0,
        experiment_name: str = "default",
    ):
        super().__init__(
            GaussianLikelihoodModel(
                experiment_name=experiment_name, model_error="additive"
            )
        )
        self.l_model = l_model
        self.tolerance = tolerance
        self.gamma = gamma
        self.bias_model = "embedded"


class EmbeddedUncorrelatedModelError(UncorrelatedModelError):
    """
    This class implements the embedded likelihood model base class.

    Parameters
    ----------
    likelihood_model_base
        An instance of EmbeddedLikelihoodBaseModel which contains general information on the
        likelihood model but no computing-methods.

    Attributes
    ----------
    bias_model
        The bias model used in this class. This is set to "embedded". Works as a flag to
        distinguish between the different bias models.
    """

    def __init__(self, likelihood_model_base: EmbeddedLikelihoodBaseModel):
        super().__init__(likelihood_model_base)
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
    """
    This class implements the moment matching (ABC) likelihood model from Sargsyan.

    Parameters
    ----------
    likelihood_model_base
        An instance of EmbeddedLikelihoodBaseModel which contains general information on the
        likelihood model but no computing-methods.

    Attributes
    ----------
    tolerance
        The tolerance value used in the moment matching likelihood model.
    gamma
        The gamma value used in the moment matching likelihood model.
    l_model
        The likelihood model used in this class. This is set to "moment_matching".
    """

    def __init__(self, likelihood_model_base: EmbeddedLikelihoodBaseModel):
        super().__init__(likelihood_model_base)
        self.tolerance = likelihood_model_base.tolerance
        self.gamma = likelihood_model_base.gamma
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
        # Uses Sargsyan2019 Eq. 15 as ABC likelihood function, corrected to include noise

        if np.isnan(response_vector).any():
            return -np.inf

        # Adjustable weights for the mean and std of the moment residuals
        if not hasattr(self, "weight_mean"):
            self.weight_mean = 1.0
        if not hasattr(self, "weight_std"):
            self.weight_std = 1.0

        # Load the standard deviations and noise values
        std_model, std_meas, stds_are_scalar = self.std_values(prms)
        variance = np.power(std_model, 2)
        n = len(residual_vector)

        #   Original likelihood
        # ll = -1 / 2 * np.log(2 * np.pi * self.tolerance**2)
        # ll -= 0.5 / self.tolerance**2 * np.sum(np.square(residual_vector)+np.square(response_vector[1]-self.gamma*np.abs(residual_vector)))

        # Noise-corrected likelihood
        if std_meas is not None:
            variance += np.power(std_meas, 2)

        # Homoscedastic noise
        if stds_are_scalar:
            std_vector = np.full_like(residual_vector, np.sqrt(variance))
            ll = -0.5 * n * np.log(2 * np.pi * self.tolerance**2 * variance)
            ll -= 0.5 * np.sum(
                np.square(self.weight_mean * residual_vector) / variance
                + np.square(
                    self.weight_std
                    * np.sqrt(np.square(response_vector[1]) + np.square(std_vector))
                    - self.gamma * np.abs(residual_vector)
                )
                / self.tolerance**2
            )

        # Heteroscedastic noise (not implemented)
        else:
            raise NotImplementedError(
                "Heteroscedastic noise is not implemented for the moment matching likelihood model."
            )

        # Store the mean and std of the moment residuals if requested
        if hasattr(self, "moment_residuals"):
            self.moment_residuals["mean"].append(np.sum(residual_vector))
            self.moment_residuals["std"].append(
                np.sum(
                    np.sqrt(np.square(response_vector[1]) + np.square(std_vector))
                    - self.gamma * np.abs(residual_vector)
                )
            )
        return ll


class GlobalMomentMatchingModelError(EmbeddedUncorrelatedModelError):
    """
    This class implements the global moment matching likelihood model.

    Parameters
    ----------
    likelihood_model_base
        An instance of EmbeddedLikelihoodBaseModel which contains general information on the
        likelihood model but no computing-methods.

    Attributes
    ----------
    gamma
        The gamma value used in the moment matching likelihood model.
    l_model
        The likelihood model used in this class. This is set to "global_moment_matching".
    """

    def __init__(self, likelihood_model_base: EmbeddedLikelihoodBaseModel):
        super().__init__(likelihood_model_base)
        self.gamma = likelihood_model_base.gamma
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
        if np.isnan(response_vector).any():
            return -np.inf

        # Load the standard deviations and noise values
        std_model, _, stds_are_scalar = self.std_values(prms)
        if stds_are_scalar:
            std_model = np.full_like(residual_vector, std_model)
        variance_noise = np.power(std_model, 2)
        n_y = len(residual_vector)

        # Calculate the intermediate statistics
        variance_population_f = np.mean(np.square(response_vector[1])) + variance_noise # Eq. 31
        mean_samples_u = np.mean(residual_vector) # Eq. 35
        variance_samples_u = np.var(residual_vector, ddof=1) #  Eq. 36, ddof=1 for sample variance

        # Calculate the log-likelihood
        ll = 0.0
        # Mean matching (L_1)
        ll -= 0.5 * np.log(2 * np.pi / n_y * variance_population_f)
        ll -= 0.5 * n_y * np.square(mean_samples_u) / variance_population_f 
        # Variance matching (L_2)
        ll -= (n_y - 1) * 0.5 * np.log(2)
        ll -= math.lgamma((n_y - 1) / 2)
        ll -= 0.5 * n_y * variance_samples_u / variance_population_f
        ll += ((n_y - 1) / 2 - 1) * np.log(n_y * variance_samples_u / variance_population_f)
        return ll
    
class RelativeGlobalMomentMatchingModelError(EmbeddedUncorrelatedModelError):
    """
    This class implements the relative error global moment matching likelihood model.

    Parameters
    ----------
    likelihood_model_base
        An instance of EmbeddedLikelihoodBaseModel which contains general information on the
        likelihood model but no computing-methods.

    Attributes
    ----------
    gamma
        The gamma value used in the moment matching likelihood model.
    l_model
        The likelihood model used in this class. This is set to "relative_global_moment_matching".
    """

    def __init__(self, likelihood_model_base: EmbeddedLikelihoodBaseModel):
        super().__init__(likelihood_model_base)
        self.gamma = likelihood_model_base.gamma
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

        if np.isnan(response_vector).any():
            return -np.inf

        # Load the standard deviations and noise values
        std_model, std_meas, stds_are_scalar = self.std_values(prms)
        variance = np.power(std_model, 2)
        n = len(residual_vector)

        # Calculate the intermediate statistics
        sigma_model_population = np.sqrt(
            self.gamma**2
            * np.square(residual_vector)
            # + variance
        )
        # sigma_model_sample = np.sqrt(np.square(response_vector[1]) + variance)
        # population_variance = (
        #     np.var(np.divide(residual_vector, sigma_model_population)) + np.mean(1 - np.divide(variance, np.square(sigma_model_population)+variance))
        # )
        # sample_variance = np.var(np.divide(residual_vector, sigma_model_sample)) + np.mean(1 - np.divide(variance, np.square(sigma_model_sample)+variance))
        # mean_residual = np.mean(np.divide(residual_vector, sigma_model_sample))

        sigma_model_sample = np.sqrt(np.square(response_vector[1]) + variance)
        # population_variance = (
        #     np.var(np.divide(residual_vector, sigma_model_population)) + 1 - np.mean(np.divide(variance, np.square(sigma_model_population)+variance))
        # )
        population_variance = 1.0
        # sample_variance = np.var(np.divide(residual_vector, sigma_model_sample)) + 1 - np.mean(np.divide(variance, np.square(sigma_model_sample)+variance))
        sample_variance = np.var(np.divide(residual_vector,sigma_model_sample))
        # mean_residual = np.mean(np.divide(residual_vector, sigma_model_sample))
        mean_residual = np.mean(np.divide(residual_vector,sigma_model_sample))
        # Calculate the log-likelihood
        ll = 0.0
        if std_meas is not None:
            variance += np.power(std_meas, 2)
        if stds_are_scalar:
            ll -= 0.5 * np.log(2 * np.pi / n * population_variance)
            ll -= 0.5 * n / population_variance * np.square(mean_residual)
            ll -= 0.5 * n * sample_variance / population_variance
            ll -= (n - 1) / 2 * np.log(2)
            ll -= math.lgamma((n - 1) / 2)
            ll += ((n - 1) / 2 - 1) * np.log(n * sample_variance / population_variance)
        return ll
class IndependentNormalModelError(EmbeddedUncorrelatedModelError):
    """
    This class implements the independent normal likelihood model.

    Parameters
    ----------
    likelihood_model_base
        An instance of EmbeddedLikelihoodBaseModel which contains general information on the
        likelihood model but no computing-methods.

    Attributes
    ----------
    l_model
        The likelihood model used in this class. This is set to "independent_normal".
    """

    def __init__(self, likelihood_model_base: EmbeddedLikelihoodBaseModel):
        super().__init__(likelihood_model_base)
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

        if np.isnan(response_vector).any():
            return -np.inf

        # Load the standard deviations and noise values
        std_model, std_meas, stds_are_scalar = self.std_values(prms)
        variance = np.power(std_model, 2)
        sigma_model_sample = np.sqrt(np.square(response_vector[1]) + variance)

        ll = 0
        if std_meas is not None:
            variance += np.power(std_meas, 2)
        if stds_are_scalar:
            ll -= 0.5 * np.sum(
                np.square(np.divide(residual_vector, sigma_model_sample))
                + np.log(2 * np.pi * np.square(sigma_model_sample))
            )
        return ll


def translate_likelihood_model(
    lm_def: EmbeddedLikelihoodBaseModel,
) -> EmbeddedUncorrelatedModelError:
    """
    Translates a given instance of EmbeddedLikelihoodBaseModel (which is essentially just a
    description of the likelihood model without any computing-methods) to a specific
    likelihood model object which does contain SciPy-based computing-methods.

    Parameters
    ----------
    lm_def
        An instance of EmbeddedLikelihoodBaseModel which contains general information on the
        likelihood model but no computing-methods.
    Returns
    -------
    likelihood_computer
        An instance of a specific likelihood model class with computing capabilities.
        Examples for such classes are given above in this file.
    """

    # likelihood model selection based on the flags given in the likelihood definition
    if not lm_def.considers_correlation:
        l_class = f"Embedded_{lm_def.l_model}_Uncorrelated"
    else:
        raise NotImplementedError

    # this dict allows to map an assigned string from the if-cases above to a specific
    # likelihood model defined in this file; the class is not assigned directly to the
    # variable l_class from the if-clauses above to avoid that l_class can have multiple
    # different types (which leads to issues during type-checking)
    class_dict = {
        "Embedded_moment_matching_Uncorrelated": MomentMatchingModelError,
        "Embedded_global_moment_matching_Uncorrelated": GlobalMomentMatchingModelError,
        "Embedded_relative_global_moment_matching_Uncorrelated": RelativeGlobalMomentMatchingModelError,
        "Embedded_independent_normal_Uncorrelated": IndependentNormalModelError,
        "Embedded_sampled_global_moment_matching_Uncorrelated": SampledGlobalMomentMatchingModelError,
        "Embedded_sampled_relative_global_moment_matching_Uncorrelated": SampledRelativeGlobalMomentMatchingModelError,
    }

    # this is where the translation happens
    likelihood_computer = class_dict[l_class](lm_def)

    return likelihood_computer
