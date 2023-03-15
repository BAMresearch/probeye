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
    ):
        logger.debug(f"Initializing {self.__class__.__name__}")
        # check that the problem does not contain a uninformative prior
        check_for_uninformative_priors(problem)
        # initialize the scipy-based solver (ScipySolver)
        super().__init__(problem, seed=seed, show_progress=show_progress)

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