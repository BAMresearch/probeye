# third party imports
from scipy import stats

# local imports
from probeye.subroutines import list2dict


class PriorTemplate:
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
            The name of the calibration-parameter the prior refers to.
        prms_def : list[str]
            A list of strings defining the prior's parameter names.
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
        self.prms_def_no_ref = list2dict(prms_def)
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


class PriorNormal(PriorTemplate):
    """Prior class for a normal distribution."""
    def __init__(self, ref_prm, prms_def, name):
        """Check out the explanations for PriorTemplate given above."""
        super().__init__(ref_prm, prms_def, name, "normal distribution")

    def __call__(self, prms, method):
        """
        Evaluates stats.norm.<method>(x, loc, scale). This function is mostly
        used with method='logpdf' during the sampling procedure.

        Parameters
        ----------
        prms : dict
            Contains the prior's parameters as keys and their values as values.
        method : string
            The method of stats.norm to be evaluated (e.g. 'pdf' or 'logpdf').

        Returns
        -------
        float
            The result of stats.norm.<method>(x, loc, scale).
        """
        x = prms[self.ref_prm]
        loc = prms[f"loc_{self.ref_prm}"]
        scale = prms[f"scale_{self.ref_prm}"]
        return getattr(stats.norm, method)(x, loc=loc, scale=scale)

    def generate_samples(self, prms, size, seed=None):
        """
        Randomly draws samples from this prior distribution. This method is used
        to create initial samples for MCMC-based algorithms.

        Parameters
        ----------
        prms : dict
            Contains the prior's parameters as keys and their values as values.
        size : int
            Number of samples to generate.
        seed : int or None, optional
            Used for the random state of the random number generation.

        Returns
        -------
        numpy.ndarray
            The generate samples.
        """
        loc = prms[f"loc_{self.ref_prm}"]
        scale = prms[f"scale_{self.ref_prm}"]
        return stats.norm.rvs(loc=loc, scale=scale, size=size,
                              random_state=seed)


class PriorLognormal(PriorTemplate):
    """Prior class for a log-normal distribution."""
    def __init__(self, ref_prm, prms_def, name):
        """Check out the explanations for PriorTemplate given above."""
        super().__init__(ref_prm, prms_def, name, "log-normal distribution")

    def __call__(self, prms, method, shape=1):
        """
        Evaluates stats.lognorm.<method>(x, loc, scale). This function is mostly
        used with method='logpdf' during the sampling procedure.

        Parameters
        ----------
        prms : dict
            Contains the prior's parameters as keys and their values as values.
        method : string
            The method of stats.lognorm to be evaluated (e.g. 'pdf', 'logpdf').
        shape : float or int
            Scipy uses this shape parameter, which is not considered as a prior
            parameter here. So, it is set to 1, which results in the standard
            version of the lognormal distribution.

        Returns
        -------
        float
            The result of stats.lognorm.<method>(x, loc, scale).
        """
        x = prms[self.ref_prm]
        loc = prms[f"loc_{self.ref_prm}"]
        scale = prms[f"scale_{self.ref_prm}"]
        return getattr(stats.lognorm, method)(x, shape, loc=loc, scale=scale)

    def generate_samples(self, prms, size, seed=None, shape=1):
        """
        Randomly draws samples from this prior distribution. This method is used
        to create initial samples for MCMC-based algorithms.

        Parameters
        ----------
        prms : dict
            Contains the prior's parameters as keys and their values as values.
        size : int
            Number of samples to generate.
        seed : int or None, optional
            Used for the random state of the random number generation.
        shape : float or int, optional
            Scipy uses a shape parameter. For the common lognormal distribution
            this shape parameter is one.

        Returns
        -------
        numpy.ndarray
            The generate samples.
        """
        loc = prms[f"loc_{self.ref_prm}"]
        scale = prms[f"scale_{self.ref_prm}"]

        return stats.lognorm.rvs(shape, loc=loc, scale=scale, size=size,
                                 random_state=seed)


class PriorUniform(PriorTemplate):
    """Prior class for a uniform distribution."""
    def __init__(self, ref_prm, prms_def, name):
        """Check out the explanations for PriorTemplate given above."""
        super().__init__(ref_prm, prms_def, name, "uniform distribution")

    def __call__(self, prms, method):
        """
        Evaluates stats.uniform.<method>(x, loc, scale). This function is mostly
        used with method='logpdf' during the sampling procedure.

        Parameters
        ----------
        prms : dict
            Contains the prior's parameters as keys and their values as values.
        method : string
            The method of stats.uniform to be evaluated (e.g. 'pdf', 'logpdf').

        Returns
        -------
        float
            The result of stats.uniform.<method>(x, loc, scale).
        """
        x = prms[self.ref_prm]
        low = prms[f"low_{self.ref_prm}"]
        high = prms[f"high_{self.ref_prm}"]
        return getattr(stats.uniform, method)(x, loc=low, scale=high-low)

    def generate_samples(self, prms, size, seed=None):
        """
        Randomly draws samples from this prior distribution. This method is used
        to create initial samples for MCMC-based algorithms.

        Parameters
        ----------
        prms : dict
            Contains the prior's parameters as keys and their values as values.
        size : int
            Number of samples to generate.
        seed : int or None, optional
            Used for the random state of the random number generation.

        Returns
        -------
        numpy.ndarray
            The generate samples.
        """
        low = prms[f"low_{self.ref_prm}"]
        high = prms[f"high_{self.ref_prm}"]
        return stats.uniform.rvs(loc=low, scale=high-low, size=size,
                                 random_state=seed)


class PriorWeibull(PriorTemplate):
    """Prior class for a three-parameter Weibull distribution."""
    def __init__(self, ref_prm, prms_def, name):
        """Check out the explanations for LogPriorTemplate given above."""
        super().__init__(ref_prm, prms_def, name, "Weibull distribution")

    def __call__(self, prms, method):
        """
        Evaluates stats.weibull_min.<method>(x, loc, scale). This function is
        mostly used with method='logpdf' during the sampling procedure.

        Parameters
        ----------
        prms : dict
            Contains the prior's parameters as keys and their values as values.
        method : string
            The method of stats.weibull_min to be evaluated (e.g. 'pdf',
            'logpdf', etc.).

        Returns
        -------
        float
            The result of stats.weibull_min.<method>(x, loc, scale).
        """
        x = prms[self.ref_prm]
        shape = prms[f"shape_{self.ref_prm}"]
        loc = prms[f"loc_{self.ref_prm}"]
        scale = prms[f"scale_{self.ref_prm}"]
        return getattr(stats.weibull_min, method)(x, shape, loc=loc,
                                                  scale=scale)

    def generate_samples(self, prms, size, seed=None):
        """
        Randomly draws samples from this prior distribution. This method is used
        to create initial samples for MCMC-based algorithms.

        Parameters
        ----------
        prms : dict
            Contains the prior's parameters as keys and their values as values.
        size : int
            Number of samples to generate.
        seed : int or None, optional
            Used for the random state of the random number generation.

        Returns
        -------
        numpy.ndarray
            The generate samples.
        """
        shape = prms[f"shape_{self.ref_prm}"]
        loc = prms[f"loc_{self.ref_prm}"]
        scale = prms[f"scale_{self.ref_prm}"]
        return stats.weibull_min.rvs(shape, loc=loc, scale=scale, size=size,
                                     random_state=seed)
