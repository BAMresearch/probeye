# third party imports
from scipy import stats

# local imports
from probeye.definition.prior import PriorBase


class PriorNormal(PriorBase):
    """Prior class for a normal distribution."""

    def __init__(self, ref_prm, prms_def, name):
        """
        Parameters
        ----------
        ref_prm : string
            The name of the latent parameter the prior refers to.
        prms_def : list
            A list of strings defining the prior's parameter names.
        name : string
            Defining the priors name.
        """
        super().__init__(ref_prm, prms_def, name, "normal distribution")

    def __call__(self, prms, method, use_ref_prm=True):
        """
        Evaluates stats.norm.<method>(x, loc, scale) or, if use_ref_prm=False
        stats.norm.<method>(loc, scale). This function is mostly used with
        method='logpdf' during the sampling procedure.

        Parameters
        ----------
        prms : dict
            Contains the prior's parameters as keys and their values as values.
        method : string
            The method of stats.norm to be evaluated (e.g. 'pdf' or 'logpdf').
        use_ref_prm : bool, optional
            If True stats.norm.<method>(x, loc, scale) is evaluated, hence 'x'
            must be provided in the prms dictionary. Otherwise, the evaluated
            method is stats.norm.<method>(loc, scale).

        Returns
        -------
        float
            The result of stats.norm.<method>(x, loc, scale) or of
            stats.norm.<method>(loc, scale).
        """
        fun = getattr(stats.norm, method)
        loc = prms[f"loc_{self.ref_prm}"]
        scale = prms[f"scale_{self.ref_prm}"]
        if use_ref_prm:
            x = prms[self.ref_prm]
            return fun(x, loc=loc, scale=scale)
        else:
            return fun(loc=loc, scale=scale)

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


class PriorLognormal(PriorBase):
    """Prior class for a log-normal distribution."""

    def __init__(self, ref_prm, prms_def, name):
        """
        Parameters
        ----------
        ref_prm : string
            The name of the latent parameter the prior refers to.
        prms_def : list
            A list of strings defining the prior's parameter names.
        name : string
            Defining the priors name.
        """
        super().__init__(ref_prm, prms_def, name, "log-normal distribution")

    def __call__(self, prms, method, shape=1, use_ref_prm=True):
        """
        Evaluates stats.lognorm.<method>(x, loc, scale) or, if use_ref_prm=False
        stats.lognorm.<method>(loc, scale). This function is mostly used with
        method='logpdf' during the sampling procedure.

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
        use_ref_prm : bool, optional
            If True stats.norm.<method>(x, loc, scale) is evaluated, hence 'x'
            must be provided in the prms dictionary. Otherwise, the evaluated
            method is stats.norm.<method>(loc, scale).

        Returns
        -------
        float
            The result of stats.lognorm.<method>(x, loc, scale) or of
            stats.lognorm.<method>(loc, scale).
        """
        fun = getattr(stats.lognorm, method)
        loc = prms[f"loc_{self.ref_prm}"]
        scale = prms[f"scale_{self.ref_prm}"]
        if use_ref_prm:
            x = prms[self.ref_prm]
            return fun(x, shape, loc=loc, scale=scale)
        else:
            return fun(shape, loc=loc, scale=scale)

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


class PriorUniform(PriorBase):
    """Prior class for a uniform distribution."""

    def __init__(self, ref_prm, prms_def, name):
        """
        Parameters
        ----------
        ref_prm : string
            The name of the latent parameter the prior refers to.
        prms_def : list
            A list of strings defining the prior's parameter names.
        name : string
            Defining the priors name.
        """
        super().__init__(ref_prm, prms_def, name, "uniform distribution")

    def __call__(self, prms, method, use_ref_prm=True):
        """
        Evaluates stats.uniform.<method>(x, loc, scale) or, if use_ref_prm=False
        stats.uniform.<method>(loc, scale). This function is mostly used with
        method='logpdf' during the sampling procedure.

        Parameters
        ----------
        prms : dict
            Contains the prior's parameters as keys and their values as values.
        method : string
            The method of stats.uniform to be evaluated (e.g. 'pdf', 'logpdf').
        use_ref_prm : bool, optional
            If True stats.norm.<method>(x, loc, scale) is evaluated, hence 'x'
            must be provided in the prms dictionary. Otherwise, the evaluated
            method is stats.norm.<method>(loc, scale).

        Returns
        -------
        float
            The result of stats.uniform.<method>(x, loc, scale) or of
            stats.uniform.<method>(loc, scale).
        """
        fun = getattr(stats.uniform, method)
        low = prms[f"low_{self.ref_prm}"]
        high = prms[f"high_{self.ref_prm}"]
        if use_ref_prm:
            x = prms[self.ref_prm]
            return fun(x, loc=low, scale=high-low)
        else:
            return fun(loc=low, scale=high - low)

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


class PriorWeibull(PriorBase):
    """Prior class for a three-parameter Weibull distribution."""

    def __init__(self, ref_prm, prms_def, name):
        """
        Parameters
        ----------
        ref_prm : string
            The name of the latent parameter the prior refers to.
        prms_def : list
            A list of strings defining the prior's parameter names.
        name : string
            Defining the priors name.
        """
        super().__init__(ref_prm, prms_def, name, "Weibull distribution")

    def __call__(self, prms, method, use_ref_prm=True):
        """
        Evaluates stats.weibull_min.<method>(x, loc, scale) or, if use_ref_prm=
        False, stats.weibull_min.<method>(loc, scale). This function is mostly
        used with method='logpdf' during the sampling procedure.

        Parameters
        ----------
        prms : dict
            Contains the prior's parameters as keys and their values as values.
        method : string
            The method of stats.weibull_min to be evaluated (e.g. 'pdf',
            'logpdf', etc.).
        use_ref_prm : bool, optional
            If True stats.weibull_min.<method>(x, loc, scale) is evaluated,
            hence 'x' must be provided in the prms dictionary. Otherwise, the
            evaluated method is weibull_min.norm.<method>(loc, scale).

        Returns
        -------
        float
            The result of stats.weibull_min.<method>(x, loc, scale) or of
            stats.weibull_min.<method>(loc, scale).
        """
        fun = getattr(stats.weibull_min, method)
        shape = prms[f"shape_{self.ref_prm}"]
        loc = prms[f"loc_{self.ref_prm}"]
        scale = prms[f"scale_{self.ref_prm}"]
        if use_ref_prm:
            x = prms[self.ref_prm]
            return fun(x, shape, loc=loc, scale=scale)
        else:
            return fun(shape, loc=loc, scale=scale)

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

def translate_prior(prior_template, prior_classes=None):
    """
    Translate a given instance of PriorBase (which is essentially just a
    description of the prior without compute-methods) to a specific prior object
    which does contain compute-methods (e.g. compute the logpdf).

    Parameters
    ----------
    prior_template : obj[PriorBase]
        An instance of PriorBase which contains basic information on the
        prior but no computing-methods.
    prior_classes : dict, None
        If None, the prior classes of defined in this file are used to translate
        the prior_template. If other user-defined priors should be used, they
        can be provided via this argument, by providing a dictionary with the
        prior_type as key and the custom prior class as value.

    Returns
    -------
    prior_object : obj[PriorBase]
        An instance of a specific prior class with computing capabilities.
        Examples for such classes are given above in this file.
    """

    # check the prior_classes argument; it either must be None, or of type dict
    if type(prior_classes) is not dict:
        if prior_classes is None:
            prior_classes = {'normal': PriorNormal,
                             'lognormal': PriorLognormal,
                             'uniform': PriorUniform,
                             'weibull': PriorWeibull}
        else:
            # in this case prior_classes is not None, and not of type dict
            raise TypeError(
                f"Custom prior_classes must be given as a dictionary. However, "
                "you provided an input of type {type(prior_classes)}.")

    # prepare the corresponding prior object; the following translation is
    # necessary, because prms_def must be given in form of a list, but was
    # already translated to a dictionary when instantiating the PriorBase
    # objects; hence prior_template.prms_def is a dictionary
    prms_def = [{key: value} for key, value in prior_template.prms_def.items()
                if key != prior_template.ref_prm]
    prior_object = prior_classes[prior_template.prior_type](
        prior_template.ref_prm, prms_def, prior_template.name)

    return prior_object
