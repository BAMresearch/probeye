# local imports
from probeye.subroutines import list2dict


class PriorBase:
    """
    Template class for prior definitions. Note that the main motivation of how
    this class is implemented was to avoid storing any numeric values for any of
    the priors parameters within the prior object.
    """
    def __init__(self, ref_prm, prms_def, name, prior_type):
        """
        Parameters
        ----------
        ref_prm : string
            The name of the latent parameter the prior refers to.
        prms_def : list
            A list of strings, or list of one-element-dicts defining the prior's
            parameter names. E.g. ['loc_a', 'scale_a'] or {'loc_a': 'loc_a',
            'scale_a': 'scale_a'}. The latter example is the notation for the
            use of global and local names, which should not be necessary for
            the definition of prior-parameters. A special case is the
            uninformative prior (see below) which hasn't got an parameters. So,
            in this case prms_def might also be None.
        name : string
            Defining the priors name.
        prior_type : string
            Stating the prior's type, e.g. "normal distribution". This is just
            used for printing information on the prior.
        """
        # write arguments to attributes
        self.ref_prm = ref_prm
        self.name = name
        self.prior_type = prior_type

        # these dictionaries contain the <global name>:<local name> pairs of the
        # prior's parameters with and without the reference variable; if for
        # example you define a normal prior for parameter 'a' with location,
        # then 'a' is the reference variable; note that the conversion to a
        # dictionary via list2dict is due to the possibility of using local
        # parameter names, which however is not intended to be used for priors
        self.hyperparameters = list2dict(prms_def)
        self.prms_def = list2dict([ref_prm] + prms_def)

    def __str__(self):
        """
        Allows printing an object of this class.

        Returns
        -------
        s : string
            A string containing details on the respective prior.
        """
        s = f"{self.prior_type} for '{self.ref_prm}', prms={self.prms_def}"
        return s
