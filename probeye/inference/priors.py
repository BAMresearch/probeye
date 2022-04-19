# standard library
from typing import Union, List, Optional

# third party imports
import numpy as np
from scipy import stats

# local imports
from probeye.definition.prior import PriorBase
from probeye.subroutines import len_or_one


class PriorNormal(PriorBase):
    """
    Prior class for a uni- or multivariate normal distribution.

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

    def __init__(
        self,
        ref_prm: str,
        prms_def: Union[str, dict, List[Union[str, dict]]],
        name: str,
    ):
        super().__init__(ref_prm, prms_def, name, "normal distribution")

    def __call__(
        self, prms: dict, method: str, use_ref_prm: bool = True, **kwargs
    ) -> Union[float, np.ndarray]:
        """
        Evaluates stats.(multivariate_)norm.<method>(x, loc, scale) or, if use_ref_prm=
        False stats.(multivariate_)norm.<method>(loc, scale). This function is mostly
        used with method='logpdf' during the sampling procedure.

        Parameters
        ----------
        prms
            Contains the prior's parameters as keys and their values as values.
        method
            The method of stats.norm to be evaluated (e.g. 'pdf' or 'logpdf').
        use_ref_prm
            If True stats.(multivariate_)norm.<method>(x, loc, scale) is evaluated,
            hence 'x' must be provided in the prms dictionary. Otherwise, the evaluated
            method is stats.norm.<method>(loc, scale).
        kwargs
            Additional keyword arguments to pass to the specified method.

        Returns
        -------
            The result of stats.(multivariate_)norm.<method>(x, loc, scale) or of
            stats.(multivariate_)norm.<method>(loc, scale).
        """
        mean = prms[f"mean_{self.ref_prm}"]
        if len_or_one(mean) == 1:
            std = prms[f"std_{self.ref_prm}"]
            fun = getattr(stats.norm, method)
            if use_ref_prm:
                x = prms[self.ref_prm]
                return fun(x, loc=mean, scale=std, **kwargs)
            else:
                return fun(loc=mean, scale=std, **kwargs)
        else:
            cov = prms[f"cov_{self.ref_prm}"]
            try:
                fun = getattr(stats.multivariate_normal, method)
            except AttributeError:
                # this try-catch construct accounts for the fact, that the multivariate
                # normal distribution does not have a 'mean' or 'median' method
                if method in ["mean", "median"]:
                    return mean
                else:
                    raise AttributeError(
                        f"stats.multivariate_normal does "
                        f"not have a '{method}'-method"
                    )
            if use_ref_prm:
                x = prms[self.ref_prm]
                return fun(x, mean=mean, cov=cov, **kwargs)
            else:
                return fun(mean=mean, cov=cov, **kwargs)

    def generate_samples(
        self, prms: dict, size: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Randomly draws samples from this prior distribution. This method is used to
        create initial samples for MCMC-based algorithms.

        Parameters
        ----------
        prms
            Contains the prior's parameters as keys and their values as values.
        size
            Number of samples to generate.
        seed
            Used for the random state of the random number generation.

        Returns
        -------
        numpy.ndarray
            The generate samples.
        """
        mean = prms[f"mean_{self.ref_prm}"]
        if len_or_one(mean) == 1:
            std = prms[f"std_{self.ref_prm}"]
            return stats.norm.rvs(loc=mean, scale=std, size=size, random_state=seed)
        else:
            cov = prms[f"cov_{self.ref_prm}"]
            return stats.multivariate_normal.rvs(
                mean=mean, cov=cov, size=size, random_state=seed
            )


class PriorLognormal(PriorBase):
    """
    Prior class for a log-normal distribution.

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

    def __init__(
        self,
        ref_prm: str,
        prms_def: Union[str, dict, List[Union[str, dict]]],
        name: str,
    ):
        super().__init__(ref_prm, prms_def, name, "log-normal distribution")

    def __call__(
        self,
        prms: dict,
        method: str,
        use_ref_prm: bool = True,
        **kwargs,
    ) -> float:
        """
        Evaluates stats.lognorm.<method>(x, loc, scale) or, if use_ref_prm=False stats.
        lognorm.<method>(loc, scale). This function is mostly used with method='logpdf'
        during the sampling procedure.

        Parameters
        ----------
        prms
            Contains the prior's parameters as keys and their values as values.
        method
            The method of stats.lognorm to be evaluated (e.g. 'pdf', 'logpdf').
        use_ref_prm
            If True stats.lognorm.<method>(x, loc, scale) is evaluated, hence 'x' must
            be provided in the prms dictionary. Otherwise, the evaluated method is
            stats.lognorm.<method>(loc, scale).
        kwargs
            Additional keyword arguments to pass to the specified method.

        Returns
        -------
            The result of stats.lognorm.<method>(x, loc, scale) or of stats.lognorm.
            <method>(loc, scale).
        """
        fun = getattr(stats.lognorm, method)
        # for understanding the following parameter-juggling check out the scipy-docs at
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
        mu = prms[f"mean_{self.ref_prm}"]
        sigma = prms[f"std_{self.ref_prm}"]
        scale = np.exp(mu)
        shape = sigma
        if use_ref_prm:
            x = prms[self.ref_prm]
            return fun(x, shape, scale=scale, **kwargs)
        else:
            return fun(shape, scale=scale, **kwargs)

    def generate_samples(
        self,
        prms: dict,
        size: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Randomly draws samples from this prior distribution. This method is used to
        create initial samples for MCMC-based algorithms.

        Parameters
        ----------
        prms
            Contains the prior's parameters as keys and their values as values.
        size
            Number of samples to generate.
        seed
            Used for the random state of the random number generation.

        Returns
        -------
            The generate samples.
        """
        # for understanding the following parameter-juggling check out the scipy-docs at
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
        mu = prms[f"mean_{self.ref_prm}"]
        sigma = prms[f"std_{self.ref_prm}"]
        scale = np.exp(mu)
        shape = sigma

        return stats.lognorm.rvs(shape, scale=scale, size=size, random_state=seed)


class PriorTruncnormal(PriorBase):
    """
    Prior class for a trunc-normal distribution.

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

    def __init__(
        self,
        ref_prm: str,
        prms_def: Union[str, dict, List[Union[str, dict]]],
        name: str,
    ):

        super().__init__(ref_prm, prms_def, name, "trunc-normal distribution")

    def __call__(
        self,
        prms: dict,
        method: str,
        use_ref_prm: bool = True,
        **kwargs,
    ) -> float:
        """
        Evaluates stats.truncnorm.<method>(x, a, b, loc, scale) or, if use_ref_prm=False
        stats.truncnorm.<method>(a, b, loc, scale). This function is mostly used with
        method='logpdf' during the sampling procedure.

        Parameters
        ----------
        prms
            Contains the prior's parameters as keys and their values as values.
        method
            The method of stats.truncnorm to be evaluated (e.g. 'pdf', 'logpdf').
        use_ref_prm
            If True stats.truncnorm.<method>(x, loc, scale) is evaluated, hence 'x' must
            be provided in the prms dictionary. Otherwise, the evaluated method is
            stats.truncnorm.<method>(loc, scale).
        kwargs
            Additional keyword arguments to pass to the specified method.

        Returns
        -------
            The result of stats.truncnorm.<method>(x, loc, scale) or of stats.truncnorm.
            <method>(loc, scale).
        """
        fun = getattr(stats.truncnorm, method)
        mean = prms[f"mean_{self.ref_prm}"]
        std = prms[f"std_{self.ref_prm}"]
        a = (prms[f"a_{self.ref_prm}"] - mean) / std
        b = (prms[f"b_{self.ref_prm}"] - mean) / std
        if use_ref_prm:
            x = prms[self.ref_prm]
            return fun(x, a=a, b=b, loc=mean, scale=std, **kwargs)
        else:
            return fun(a=a, b=b, loc=mean, scale=std, **kwargs)

    def generate_samples(
        self,
        prms: dict,
        size: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Randomly draws samples from this prior distribution. This method is used to
        create initial samples for MCMC-based algorithms.

        Parameters
        ----------
        prms
            Contains the prior's parameters as keys and their values as values.
        size
            Number of samples to generate.
        seed
            Used for the random state of the random number generation.

        Returns
        -------
            The generate samples.
        """
        mean = prms[f"mean_{self.ref_prm}"]
        std = prms[f"std_{self.ref_prm}"]
        a = (prms[f"a_{self.ref_prm}"] - mean) / std
        b = (prms[f"b_{self.ref_prm}"] - mean) / std
        return stats.truncnorm.rvs(
            a=a, b=b, loc=mean, scale=std, size=size, random_state=seed
        )


class PriorUniform(PriorBase):
    """
    Prior class for a uniform distribution.

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

    def __init__(
        self,
        ref_prm: str,
        prms_def: Union[str, dict, List[Union[str, dict]]],
        name: str,
    ):

        super().__init__(ref_prm, prms_def, name, "uniform distribution")

    def __call__(
        self, prms: dict, method: str, use_ref_prm: bool = True, **kwargs
    ) -> float:
        """
        Evaluates stats.uniform.<method>(x, loc, scale) or, if use_ref_prm=False stats.
        uniform.<method>(loc, scale). This function is mostly used with method='logpdf'
        during the sampling procedure.

        Parameters
        ----------
        prms
            Contains the prior's parameters as keys and their values as values.
        method
            The method of stats.uniform to be evaluated (e.g. 'pdf', 'logpdf').
        use_ref_prm
            If True stats.uniform.<method>(x, loc, scale) is evaluated, hence 'x' must
            be provided in the prms dictionary. Otherwise, the evaluated method is
            stats.uniform.<method>(loc, scale).
        kwargs
            Additional keyword arguments to pass to the specified method.

        Returns
        -------
            The result of stats.uniform.<method>(x, loc, scale) or of stats.uniform.
            <method>(loc, scale).
        """
        fun = getattr(stats.uniform, method)
        low = prms[f"low_{self.ref_prm}"]
        high = prms[f"high_{self.ref_prm}"]
        if use_ref_prm:
            x = prms[self.ref_prm]
            return fun(x, loc=low, scale=high - low, **kwargs)
        else:
            return fun(loc=low, scale=high - low, **kwargs)

    def generate_samples(
        self, prms: dict, size: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Randomly draws samples from this prior distribution. This method is used to
        create initial samples for MCMC-based algorithms.

        Parameters
        ----------
        prms
            Contains the prior's parameters as keys and their values as values.
        size
            Number of samples to generate.
        seed
            Used for the random state of the random number generation.

        Returns
        -------
            The generate samples.
        """
        low = prms[f"low_{self.ref_prm}"]
        high = prms[f"high_{self.ref_prm}"]
        return stats.uniform.rvs(
            loc=low, scale=high - low, size=size, random_state=seed
        )


class PriorWeibull(PriorBase):
    """
    Prior class for a three-parameter Weibull distribution.

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

    def __init__(
        self,
        ref_prm: str,
        prms_def: Union[str, dict, List[Union[str, dict]]],
        name: str,
    ):

        super().__init__(ref_prm, prms_def, name, "Weibull distribution")

    def __call__(
        self, prms: dict, method: str, use_ref_prm: bool = True, **kwargs
    ) -> float:
        """
        Evaluates stats.weibull_min.<method>(x, loc, scale) or, if use_ref_prm=False,
        stats.weibull_min.<method>(loc, scale). This function is mostly used with
        method='logpdf' during the sampling procedure.

        Parameters
        ----------
        prms
            Contains the prior's parameters as keys and their values as values.
        method
            The method of stats.weibull_min to be evaluated ('pdf', 'logpdf', etc.).
        use_ref_prm
            If True stats.weibull_min.<method>(x, loc, scale) is evaluated, hence 'x'
            must be provided in the prms dictionary. Otherwise, the evaluated method is
            weibull_min.<method>(loc, scale).
        kwargs
            Additional keyword arguments to pass to the specified method.

        Returns
        -------
            The result of stats.weibull_min.<method>(x, loc, scale) or of stats.
            weibull_min.<method>(loc, scale).
        """
        fun = getattr(stats.weibull_min, method)
        shape = prms[f"shape_{self.ref_prm}"]
        loc = prms[f"loc_{self.ref_prm}"]
        scale = prms[f"scale_{self.ref_prm}"]
        if use_ref_prm:
            x = prms[self.ref_prm]
            return fun(x, shape, loc=loc, scale=scale, **kwargs)
        else:
            return fun(shape, loc=loc, scale=scale, **kwargs)

    def generate_samples(
        self, prms: dict, size: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Randomly draws samples from this prior distribution. This method is used to
        create initial samples for MCMC-based algorithms.

        Parameters
        ----------
        prms
            Contains the prior's parameters as keys and their values as values.
        size
            Number of samples to generate.
        seed
            Used for the random state of the random number generation.

        Returns
        -------
            The generate samples.
        """
        shape = prms[f"shape_{self.ref_prm}"]
        loc = prms[f"loc_{self.ref_prm}"]
        scale = prms[f"scale_{self.ref_prm}"]
        return stats.weibull_min.rvs(
            shape, loc=loc, scale=scale, size=size, random_state=seed
        )


def translate_prior(
    prior_template: PriorBase, prior_classes: Optional[dict] = None
) -> PriorBase:
    """
    Translate a given instance of PriorBase (which is essentially just a description of
    the prior without compute-methods) to a specific prior object which does contain
    compute-methods (e.g. compute the logpdf).

    Parameters
    ----------
    prior_template
        An instance of PriorBase which contains basic information on the prior but no
        computing-methods.
    prior_classes
        If None, the prior classes of defined in this file are used to translate the
        prior_template. If other user-defined priors should be used, they can be
        provided via this argument, by providing a dictionary with the prior_type as
        key and the custom prior class as value.

    Returns
    -------
    prior_object
        An instance of a specific prior class with computing capabilities. Examples for
        such classes are given above in this file.
    """

    # no translation is required for an uninformative prior
    if prior_template.prior_type == "uninformative":
        return prior_template

    # check the prior_classes argument; it either must be None, or of type dict
    if type(prior_classes) is not dict:
        if prior_classes is None:
            prior_classes = {
                "normal": PriorNormal,
                "lognormal": PriorLognormal,
                "truncnormal": PriorTruncnormal,
                "uniform": PriorUniform,
                "weibull": PriorWeibull,
            }
        else:
            # in this case prior_classes is not None, and not of type dict
            raise TypeError(
                f"Custom prior_classes must be given as a dictionary. However, "
                f"you provided an input of type {type(prior_classes)}."
            )

    # prepare the corresponding prior object; the following translation is necessary,
    # because prms_def must be given in form of a list, but was already translated to
    # a dictionary when instantiating the PriorBase objects; hence prior_template.
    # prms_def is a dictionary
    prms_def = [
        {key: value}
        for key, value in prior_template.prms_def.items()
        if key != prior_template.ref_prm
    ]
    prior_object = prior_classes[prior_template.prior_type](
        prior_template.ref_prm, prms_def, prior_template.name
    )

    return prior_object
