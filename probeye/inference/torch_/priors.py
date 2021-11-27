# standard library imports
from typing import Union, List, Dict

# third party imports
import pyro
import pyro.distributions as dist
import torch as th

# local imports
from probeye.definition.prior import PriorBase
from probeye.subroutines import len_or_one


class PriorNormal(PriorBase):
    """Prior class for a normal distribution."""

    def __init__(
        self,
        ref_prm: str,
        prms_def: Union[str, dict, List[Union[str, dict]]],
        name: str,
    ):
        """
        Parameters
        ----------
        ref_prm
            The name of the latent parameter the prior refers to.
        prms_def
            Defines the prior's parameter names. See the docstring of PriorBase for
            more detailed information.
        name
            Defining the priors name.
        """
        super().__init__(ref_prm, prms_def, name, "normal distribution")

    def sample(self, prms: dict) -> pyro.sample:
        """
        Samples from a (multivariate-)normal distribution.

        Parameters
        ----------
        prms
            Contains the prior's parameters as keys and their values as values.

        Returns
        -------
            The generated sample.
        """
        loc = prms[f"loc_{self.ref_prm}"]
        scale = prms[f"scale_{self.ref_prm}"]
        if len_or_one(loc) == 1:
            distribution = dist.Normal(loc, scale)
        else:
            distribution = dist.MultivariateNormal(
                th.from_numpy(loc), covariance_matrix=th.from_numpy(scale)
            )
        return pyro.sample(self.ref_prm, distribution)


class PriorUniform(PriorBase):
    """Prior class for a uniform distribution."""

    def __init__(
        self,
        ref_prm: str,
        prms_def: Union[str, dict, List[Union[str, dict]]],
        name: str,
    ):
        """
        Parameters
        ----------
        ref_prm
            The name of the latent parameter the prior refers to.
        prms_def
            Defines the prior's parameter names. See the docstring of PriorBase for
            more detailed information.
        name
            Defining the priors name.
        """
        super().__init__(ref_prm, prms_def, name, "uniform distribution")

    def sample(self, prms: dict) -> pyro.sample:
        """
        Samples from a (multivariate-)normal distribution.

        Parameters
        ----------
        prms
            Contains the prior's parameters as keys and their values as values.

        Returns
        -------
            The generated sample.
        """
        low = prms[f"low_{self.ref_prm}"]
        high = prms[f"high_{self.ref_prm}"]
        return pyro.sample(self.ref_prm, dist.Uniform(low, high))


def translate_prior_template(prior_template: PriorBase) -> PriorBase:
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

    # this dictionary contains the classes from above in this file
    prior_classes = {"normal": PriorNormal, "uniform": PriorUniform}

    # prepare the corresponding prior object; the following translation is necessary,
    # because prms_def must be given in form of a list, but was already translated to a
    # dictionary when instantiating the PriorBase objects; hence prior_template.prms_def
    # is a dictionary
    prms_def = [
        {key: value}
        for key, value in prior_template.prms_def.items()
        if key != prior_template.ref_prm
    ]
    prior_object = prior_classes[prior_template.prior_type](
        prior_template.ref_prm, prms_def, prior_template.name  # type: ignore
    )

    return prior_object
