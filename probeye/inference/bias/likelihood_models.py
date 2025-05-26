# third party imports
import numpy as np
import math

# local imports
from probeye.definition.likelihood_model import GaussianLikelihoodModel

from probeye.inference.scipy.likelihood_models import (
    ScipyLikelihoodBase,
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


class MomentMatchingModelError(EmbeddedLikelihoodBaseModel):
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

    def __init__(self, tolerance: float, gamma: float, experiment_name: str):
        super().__init__(
            l_model="moment_matching",
            tolerance=tolerance,
            gamma=gamma,
            experiment_name=experiment_name,
        )

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
        # This should be set in the likelihood definition
        # Functionality is implemented but not used in the current implementation
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
        # This is used for debugging and analysis purposes
        if hasattr(self, "moment_residuals"):
            self.moment_residuals["mean"].append(np.sum(residual_vector))
            self.moment_residuals["std"].append(
                np.sum(
                    np.sqrt(np.square(response_vector[1]) + np.square(std_vector))
                    - self.gamma * np.abs(residual_vector)
                )
            )
        return ll

class IndependentNormalModelError(EmbeddedLikelihoodBaseModel):
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

    def __init__(self, experiment_name: str):
        super().__init__(
            l_model="independent_normal",
            experiment_name=experiment_name,
        )

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
        else:
            raise NotImplementedError("Heteroscedastic noise not implemented")

        return ll

class GlobalMomentMatchingModelError(EmbeddedLikelihoodBaseModel):
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

    def __init__(self, gamma: float, experiment_name: str):
        super().__init__(
            l_model="global_moment_matching",
            gamma=gamma,
            experiment_name=experiment_name,
        )

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


class RelativeGlobalMomentMatchingModelError(EmbeddedLikelihoodBaseModel):
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

    def __init__(self, gamma: float, experiment_name: str):
        super().__init__(
            l_model="relative_global_moment_matching",
            gamma=gamma,
            experiment_name=experiment_name,
        )

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
        relative_residual_vector = np.divide(residual_vector,np.sqrt(np.square(response_vector[1]) + variance_noise))

        # Calculate the intermediate statistics
        variance_population_f_r = 1.0 # Eq. 31
        mean_samples_u_r = np.mean(relative_residual_vector) # Eq. 35
        variance_samples_u_r = np.var(relative_residual_vector, ddof=1) #  Eq. 36, ddof=1 for sample variance

        # Calculate the log-likelihood
        ll = 0.0
        # Mean matching (L_1)
        ll -= 0.5 * np.log(2 * np.pi / n_y * variance_population_f_r)
        ll -= 0.5 * n_y * np.square(mean_samples_u_r) / variance_population_f_r 
        # Variance matching (L_2)
        ll -= (n_y - 1) * 0.5 * np.log(2)
        ll -= math.lgamma((n_y - 1) / 2)
        ll -= 0.5 * n_y * variance_samples_u_r / variance_population_f_r
        ll += ((n_y - 1) / 2 - 1) * np.log(n_y * variance_samples_u_r / variance_population_f_r)
        return ll
