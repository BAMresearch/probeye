# standard library
from typing import Union, List

# local imports
from probeye.definition.distribution import ProbabilityDistribution
from probeye.subroutines import translate_prms_def


class PriorBase:
    """
    Template class for prior definitions. Note that the main motivation of how this
    class is implemented was to avoid storing any numeric values for any of the priors
    parameters within the prior object.

    Parameters
    ----------
    ref_prm
        The name of the latent parameter the prior refers to.
    prms_def
        A list of strings, or list of one-element-dicts defining the prior's parameter
        names. For example ['mean_a', 'std_a'] or {'mean_a': 'std_a', 'std_a': 'std_a'}.
        The latter example is the notation for the use of global and local names, which
        should not be necessary for the definition of prior-parameters. A special case
        is the uninformative prior (see below) which hasn't got an parameters. So, in
        this case prms_def might also be None.
    name
        Defining the priors name.
    dist
        The probability distribution that describes the prior.
    """

    def __init__(
        self,
        ref_prm: str,
        prms_def: Union[str, dict, List[Union[str, dict]]],
        name: str,
        dist: ProbabilityDistribution,
    ):

        # write arguments to attributes
        self.ref_prm = ref_prm
        self.name = name
        self.dist = dist

        # these dictionaries contain the <global name>:<local name> pairs of the prior's
        # parameters with and without the reference variable; if for example you define
        # a normal prior for parameter 'a' with location, then 'a' is the reference
        # variable; note that the conversion to a dictionary via list2dict is due to the
        # possibility of using local parameter names, which however is not intended to
        # be used for priors
        self.hyperparameters, _ = translate_prms_def(prms_def)
        self.prms_def, _ = translate_prms_def(
            [ref_prm] + [*self.hyperparameters.values()]
        )

    @property
    def prior_type(self) -> str:
        """Dynamically accesses the prior's distribution type from its dist-object."""
        return self.dist.dist_type

    def __str__(self) -> str:
        """
        Allows printing an object of this class.

        Returns
        -------
        s
            A string containing details on the respective prior.
        """
        s = f"{self.prior_type} for '{self.ref_prm}', prms={self.prms_def}"
        return s
