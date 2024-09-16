# standard library imports
from typing import Optional, TYPE_CHECKING
import copy as cp

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.inverse_problem import InverseProblem


class Solver:
    """
    Base class for the different solvers (inference engines).

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

        # attributes from arguments
        self.problem = cp.copy(problem)
        self.show_progress = show_progress
        self.seed = seed

        # prepare the given problem for this solver
        self._translate_parameters()
        self._translate_experiments()
        self._translate_forward_models()
        self._translate_likelihood_models()

        # the following attributes will be set after the solver was run
        self.raw_results = None
        self.summary = {}  # type: dict

    def _translate_parameters(self):
        """
        Translate the inverse problem's parameters as needed for this solver.
        """
        raise NotImplementedError(
            f"The '_translate_parameters'-method has not been implemented yet!"
        )

    def _translate_experiments(self):
        """
        Translate the inverse problem's experiments as needed for this solver.
        """
        raise NotImplementedError(
            f"The '_translate_experiments'-method has not been implemented yet!"
        )

    def _translate_forward_models(self):
        """
        Translate the inverse problem's forward models as needed for this solver.
        """
        raise NotImplementedError(
            f"The '_translate_forward_models'-method has not been implemented yet!"
        )

    def _translate_likelihood_models(self):
        """
        Translate the inverse problem's likelihood models as needed for this solver.
        """
        raise NotImplementedError(
            f"The '_translate_likelihood_models'-method has not been implemented yet!"
        )
