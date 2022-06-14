# standard library
from typing import Union, Optional, TYPE_CHECKING

# third party imports
import numpy as np

# local imports
from probeye.definition.distribution import ProbabilityDistribution
from probeye.definition.prior import PriorBase
from probeye.inference.scipy.distributions import translate_probability_distribution

# this is required to prevent a circular import-structure
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.parameter import Parameters
    import matplotlib.pyplot as plt


class Prior(PriorBase):
    def __init__(
        self,
        ref_prm: str,
        prms_def: Union[str, dict, list],
        name: str,
        dist: ProbabilityDistribution,
    ):
        """
        This class is an extension of PriorBase with computing and plot methods.
        For details on the args check out :class:`~probeye.definition.prior.PriorBase`.
        """
        dist_with_computing = translate_probability_distribution(dist)
        super().__init__(ref_prm, prms_def, name, dist_with_computing)

    def __call__(
        self, prms: dict, method: str, use_ref_prm: bool = True, **kwargs
    ) -> Union[float, np.ndarray]:
        """
        Evaluates a given method of the underlying probability distribution. For more
        information on the arguments check out the docstring of the reference method
        :func:`~probeye.definition.distribution.ProbabilityDistribution.__call__`.
        """
        return self.dist(self.ref_prm, prms, method, use_ref_prm, **kwargs)

    def generate_samples(
        self, prms: dict, size: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates random samples from the underlying probability distribution.

        Parameters
        ----------
        prms
            Contains the prior's parameters as keys and their values as values.
        size
            The number of samples to generate.
        seed
            Used for the random state of the random number generation.
        """
        samples = self.dist(
            self.ref_prm, prms, "rvs", False, size=size, random_state=seed
        )
        assert type(samples) == np.ndarray  # this is for the benefit of mypy
        return samples

    def plot(
        self,
        ax: "plt.Axes",
        prms: "Parameters",
        x: Optional[np.ndarray] = None,
        n_points: int = 200,
        color: str = "darkorange",
        rotate: bool = False,
    ):
        """
        Plots the prior-pdf to a given axis object using the prior's 'dist'-object.
        For more information on the arguments of this method checkout the reference
        method :func:`~probeye.definition.distribution.ProbabilityDistribution.plot`.
        """
        self.dist.plot(self.ref_prm, ax, prms, x, n_points, color, rotate)


def translate_prior(prior_template: PriorBase) -> PriorBase:
    """
    Translate a given instance of PriorBase (which is essentially just a description of
    the prior without compute-methods) to a specific prior object which does contain
    compute-methods (e.g. compute the logpdf).

    Parameters
    ----------
    prior_template
        An instance of PriorBase which contains basic information on the prior but no
        computing-methods.

    Returns
    -------
    prior_object
        An instance of a specific prior class with computing capabilities. Examples for
        such classes are given above in this file.
    """

    # no translation is required for an uninformative prior
    if prior_template.prior_type == "uninformative":
        return prior_template

    # prepare the corresponding prior object; the following translation is necessary,
    # because prms_def must be given in form of a list, but was already translated to
    # a dictionary when instantiating the PriorBase objects; hence prior_template.
    # prms_def is a dictionary
    prms_def = [
        {key: value}
        for key, value in prior_template.prms_def.items()
        if key != prior_template.ref_prm
    ]
    prior_object = Prior(
        prior_template.ref_prm,
        prms_def,
        prior_template.name,
        prior_template.dist,
    )

    return prior_object
