# standard library imports
from typing import TYPE_CHECKING, Optional

# third party imports
import numpy as np
import arviz as az
from loguru import logger

# local imports
from probeye.subroutines import check_for_uninformative_priors
from probeye.inference.emcee.solver import EmceeSolver
from probeye.inference.koh.likelihood_models import translate_likelihood_model

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.inverse_problem import InverseProblem


class KOHSolver(EmceeSolver):
    """
    Provides a modification of emcee solver to work with Kennedy and O'Hagan's 
    type of implementations.

    Parameters
    ----------
    problem
        Describes the inverse problem including e.g. parameters and data.
    seed
        Random state used for random number generation.
    show_progress
        When True, the progress of a solver routine will be shown (for example as a
        progress-bar) if such a feature is available. Otherwise, the progress will
        not shown.
    """

    def __init__(
        self,
        problem: "InverseProblem",
        seed: Optional[int] = None,
        show_progress: bool = True,
        extended_problem: bool = False,
    ):
        logger.debug(f"Initializing {self.__class__.__name__}")
        # check that the problem does not contain a uninformative prior
        check_for_uninformative_priors(problem)
        # initialize the scipy-based solver (ScipySolver)
        super().__init__(problem, seed=seed, show_progress=show_progress)
        self.extended_problem = extended_problem

    def _translate_likelihood_models(self):
        """
        Translate the inverse problem's likelihood models as needed for this solver.
        """

        logger.debug("Translate the problem's likelihood models")
        for like_name in self.problem.likelihood_models:

            # the likelihood model's forward model is still referencing the old (i.e.,
            # not-translated) forward model and needs to be reset to the updated one
            fwd_name = self.problem.likelihood_models[like_name].forward_model.name
            fwd_model = self.problem.forward_models[fwd_name]
            self.problem.likelihood_models[like_name].forward_model = fwd_model
            self.problem.likelihood_models[like_name].determine_output_lengths()

            # translate the likelihood model
            self.problem.likelihood_models[like_name] = translate_likelihood_model(
                self.problem.likelihood_models[like_name]
            )
    
    def _translate_parameters(self):
        """
        Translate the inverse problem's parameters as needed for this solver.
        """
        logger.debug("Translate the problem's parameters")
        # translate the parameters
        for par_name in self.problem.parameters:
            self.problem.parameters[par_name] = self.problem.parameters[par_name].translate()

    def loglike(self, theta: np.ndarray) -> float:
        """
        Evaluates the log-likelihood function of the problem at theta.

        Parameters
        ----------
        theta
            A numeric vector for which the log-likelihood function should be evaluated.
            Which parameters these numbers refer to can be checked by calling self.
            theta_explanation() once the problem is set up.

        Returns
        -------
        ll
            The evaluated log-likelihood function for the given theta-vector.
        """

        # check whether the values of the latent parameters are within their domains;
        # they can end up outside their domains for example during sampling, when a
        # parameter vector is proposed, that contains a value that is not within the
        # specified bounds of a parameter

        if not self.problem.check_parameter_domains(theta):
            return -np.inf
        
        # Formualtion for extended problem
        if self.extended_problem:
            ll = 0.0
            for likelihood_model in self.problem.likelihood_models.values():
                # compute the model response/residuals for the likelihood model's experiment
                response, residuals = self.evaluate_model_response(
                    theta, likelihood_model.forward_model, likelihood_model.experiment_name
                )
                prms_likelihood = self.problem.get_parameters(
                    theta, likelihood_model.prms_def
                )
            return ll
        
        # Formulation for standard problem
        else:
            # compute the contribution to the log-likelihood function for each likelihood
            # model and sum it all up
            ll = 0.0
            for likelihood_model in self.problem.likelihood_models.values():
                # compute the model response/residuals for the likelihood model's experiment
                response, residuals = self.evaluate_model_response(
                    theta, likelihood_model.forward_model, likelihood_model.experiment_name
                )
                # get the parameter values for the likelihood model's parameters
                prms_likelihood = self.problem.get_parameters(
                    theta, likelihood_model.prms_def
                )
                # evaluate the loglike-contribution for the likelihood model
                ll += likelihood_model.loglike(response, residuals, prms_likelihood)
            return ll