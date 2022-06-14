# standard library
from typing import Union, Optional, TYPE_CHECKING

# third party imports
import numpy as np
from scipy import stats

# local imports
from probeye.definition.distribution import (
    ProbabilityDistribution,
    Normal,
    MultivariateNormal,
    TruncNormal,
    LogNormal,
    Uniform,
    Weibull,
    SampleBased,
)

# this is required to prevent a circular import-structure
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.parameter import Parameters
    import matplotlib.pyplot as plt


class NormalDistribution(Normal):
    """
    Normal or Gaussian probability distribution (univariate) with computing routines.
    For more information on the used computing routines check out
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html.
    """

    def __init__(self, prm_dict: dict):
        super().__init__(**prm_dict)

    @staticmethod
    def get_hyperparameters(prms: dict, primary_var: str) -> tuple:
        """
        Extracts the hyperparameters of this distribution from 'prms' and returns them.
        For more information on the arguments check out this class's
        :func:`~probeye.inference.distributions.NormalDistribution.__call__` method.
        """
        mean = prms[f"mean_{primary_var}"]
        std = prms[f"std_{primary_var}"]
        return mean, std

    def __call__(
        self,
        primary_var: str,
        prms: dict,
        method: str,
        use_primary_var: bool = True,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Evaluates scipy.stats.norm.<method>(x, loc, scale) or, if use_primary_var=False,
        scipy.stats.norm.<method>(loc, scale). This function is mostly used with
        method='logpdf' during the sampling procedure.

        Parameters
        ----------
        primary_var
            The primary variable of this distribution. This should mean the variable
            the distribution refers to.
        prms
            Contains the distribution's parameter names as keys and their numeric values
            as the dictionary's values.
        method
            The method of scipy.stats.norm to be evaluated (e.g. 'pdf' or 'logpdf').
        use_primary_var
            If True scipy.stats.norm.<method>(x, loc, scale) is evaluated, hence 'x'
            must be provided in the prms dictionary (the parameter does not need to be
            called 'x'). Otherwise, the evaluated method is without the primary variable
            hence, scipy.stats.norm.<method>(loc, scale).
        kwargs
            Additional keyword arguments to pass to the specified method.

        Returns
        -------
            The result of scipy.stats.norm.<method>(x, loc, scale) or of
            scipy.stats.norm.<method>(loc, scale).
        """
        mean, std = self.get_hyperparameters(prms, primary_var)
        fun = getattr(stats.norm, method)
        if use_primary_var:
            x = prms[primary_var]
            return fun(x, loc=mean, scale=std, **kwargs)
        else:
            return fun(loc=mean, scale=std, **kwargs)

    def plot(
        self,
        primary_var: str,
        ax: "plt.Axes",
        parameters: "Parameters",
        x: Optional[np.ndarray] = None,
        n_points: int = 200,
        color: str = "darkorange",
        rotate: bool = False,
        label="pdf",
    ):
        """
        Plots the prior's probability density function (pdf) to a given axis object.
        For more information on the arguments of this method checkout the parent class's
        :func:`~probeye.definition.distribution.ProbabilityDistribution.plot` method.
        """
        prms = parameters.value_dict
        mean, std = self.get_hyperparameters(prms, primary_var)
        # proceed, only if both values are constants and not latent parameters
        if (mean is not None) and (std is not None):
            if x is None:
                left = mean - 2.0 * std
                right = mean + 2.0 * std
                x = np.linspace(left, right, n_points)
            else:
                left = min(x)
                right = max(x)
            prms[primary_var] = x
            y = self(primary_var, prms, "pdf")
            margin = 0.025 * abs(right - left)
            adjusted_left = left - margin
            adjusted_right = right + margin
            self._plot(x, y, ax, color, rotate, adjusted_left, adjusted_right, label)


class MultivariateNormalDistribution(MultivariateNormal):
    """
    Normal or Gaussian probability distribution (multivariate) with computing routines.
    For more information on the used computing routines check out
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html.
    """

    def __init__(self, prm_dict: dict):
        super().__init__(**prm_dict)

    @staticmethod
    def get_hyperparameters(prms: dict, primary_var: str) -> tuple:
        """
        Extracts the hyperparameters of this distribution from 'prms' and returns them.
        For more information on the arguments check out this class's
        :func:`~probeye.inference.distributions.MultivariateNormalDistribution.__call__`
        method.
        """
        mean = prms[f"mean_{primary_var}"]
        std = prms[f"cov_{primary_var}"]
        return mean, std

    def __call__(
        self,
        primary_var: str,
        prms: dict,
        method: str,
        use_primary_var: bool = True,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Evaluates scipy.stats.multivariate_norm.<method>(x, loc, scale) or, if
        use_primary_var=False, scipy.stats.multivariate_norm.<method>(loc, scale). This
        function is mostly used with method='logpdf' during the sampling procedure.

        Parameters
        ----------
        primary_var
            The primary variable of this distribution. This should mean the variable
            the distribution refers to.
        prms
            Contains the distribution's parameter names as keys and their numeric values
            as the dictionary's values.
        method
            The method of scipy.stats.multivariate_norm to be evaluated (e.g. 'logpdf').
        use_primary_var
            If True scipy.stats.multivariate_norm.<method>(x, loc, scale) is evaluated,
            hence 'x' must be provided in the prms dictionary (the parameter does not
            need to be called 'x'). Otherwise, the evaluated method is without the
            primary variable hence, scipy.stats.multivariate_norm.<method>(loc, scale).
        kwargs
            Additional keyword arguments to pass to the specified method.

        Returns
        -------
            The result of scipy.stats.multivariate_norm.<method>(x, loc, scale) or of
            scipy.stats.multivariate_norm.<method>(loc, scale).
        """
        mean, cov = self.get_hyperparameters(prms, primary_var)
        try:
            fun = getattr(stats.multivariate_normal, method)
        except AttributeError:
            # this try-catch construct accounts for the fact, that the multivariate
            # normal distribution does not have a 'mean' or 'median' method
            if method in ["mean", "median"]:
                return mean
            else:
                raise AttributeError(
                    f"scipy.stats.multivariate_normal does "
                    f"not have the requested "
                    f"'{method}'-method."
                )
        if use_primary_var:
            x = prms[primary_var]
            return fun(x, mean=mean, cov=cov, **kwargs)
        else:
            return fun(mean=mean, cov=cov, **kwargs)


class TruncNormalDistribution(TruncNormal):
    """
    Truncated normal or Gaussian probability distribution (univariate) with computing
    routines. For more information on the used computing routines check out
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html.
    """

    def __init__(self, prm_dict: dict):
        super().__init__(**prm_dict)

    @staticmethod
    def get_hyperparameters(prms: dict, primary_var: str) -> tuple:
        """
        Extracts the hyperparameters of this distribution from 'prms' and returns them.
        For more information on the arguments check out this class's
        :func:`~probeye.inference.distributions.TruncNormalDistribution.__call__`
        method.
        """
        mean = prms[f"mean_{primary_var}"]
        std = prms[f"std_{primary_var}"]
        low = prms[f"low_{primary_var}"]
        high = prms[f"high_{primary_var}"]
        return mean, std, low, high

    def __call__(
        self,
        primary_var: str,
        prms: dict,
        method: str,
        use_primary_var: bool = True,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Evaluates scipy.stats.truncnorm.<method>(x, a, b, loc, scale) or, if
        use_primary_var=False, scipy.stats.truncnorm.<method>(a, b, loc, scale). This
        function is mostly used with method='logpdf' during the sampling procedure.

        Parameters
        ----------
        primary_var
            The primary variable of this distribution. This should mean the variable
            the distribution refers to.
        prms
            Contains the distribution's parameter names as keys and their numeric values
            as the dictionary's values.
        method
            The method of scipy.stats.truncnorm to be evaluated (e.g. 'logpdf').
        use_primary_var
            If True scipy.stats.truncnorm.<method>(x, a, b, loc, scale) is evaluated,
            hence 'x' must be provided in the prms dictionary (the parameter does not
            need to be called 'x'). Otherwise, the evaluated method is without the
            primary variable hence, scipy.stats.truncnorm.<method>(a, b, loc, scale).
        kwargs
            Additional keyword arguments to pass to the specified method.

        Returns
        -------
            The result of scipy.stats.truncnorm.<method>(x, a, b, loc, scale) or of
            scipy.stats.truncnorm.<method>(a, b, loc, scale).
        """
        mean, std, low, high = self.get_hyperparameters(prms, primary_var)
        # the following transformation is required to be consistent with stats.truncnorm
        # which expects the bounds to be defined with respect to the std. normal dist.
        a = (low - mean) / std
        b = (high - mean) / std
        fun = getattr(stats.truncnorm, method)
        if use_primary_var:
            x = prms[primary_var]
            return fun(x, a=a, b=b, loc=mean, scale=std, **kwargs)
        else:
            return fun(a=a, b=b, loc=mean, scale=std, **kwargs)

    def plot(
        self,
        primary_var: str,
        ax: "plt.Axes",
        parameters: "Parameters",
        x: Optional[np.ndarray] = None,
        n_points: int = 200,
        color: str = "darkorange",
        rotate: bool = False,
        label="pdf",
    ):
        """
        Plots the prior's probability density function (pdf) to a given axis object.
        For more information on the arguments of this method checkout the parent class's
        :func:`~probeye.definition.distribution.ProbabilityDistribution.plot` method.
        """
        prms = parameters.value_dict
        mean, std, low, high = self.get_hyperparameters(prms, primary_var)
        # proceed, only if both values are constants and not latent parameters
        if (mean is not None) and (std is not None):
            if x is None:
                x = np.linspace(low, high, n_points)
            prms[primary_var] = x
            y = self(primary_var, prms, "pdf")
            margin = 0.025 * abs(low - high)
            adjusted_left = low - margin
            adjusted_right = high + margin
            self._plot(x, y, ax, color, rotate, adjusted_left, adjusted_right, label)


class LogNormalDistribution(LogNormal):
    """
    Log-normal probability distribution (univariate) with computing routines. For more
    information on the used computing routines check out
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html.
    """

    def __init__(self, prm_dict: dict):
        super().__init__(**prm_dict)

    @staticmethod
    def get_hyperparameters(prms: dict, primary_var: str) -> tuple:
        """
        Extracts the hyperparameters of this distribution from 'prms' and returns them.
        For more information on the arguments check out this class's
        :func:`~probeye.inference.distributions.LogNormalDistribution.__call__` method.
        """
        mean = prms[f"mean_{primary_var}"]
        std = prms[f"std_{primary_var}"]
        return mean, std

    def __call__(
        self,
        primary_var: str,
        prms: dict,
        method: str,
        use_primary_var: bool = True,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Evaluates scipy.stats.lognorm.<method>(x, s, scale, loc=0) or, if
        use_primary_var=False, scipy.stats.lognorm.<method>(s, scale, loc=0). This
        function is mostly used with method='logpdf' during the sampling procedure.

        Parameters
        ----------
        primary_var
            The primary variable of this distribution. This should mean the variable
            the distribution refers to.
        prms
            Contains the distribution's parameter names as keys and their numeric values
            as the dictionary's values.
        method
            The method of scipy.stats.lognorm to be evaluated (e.g. 'pdf' or 'logpdf').
        use_primary_var
            If True scipy.stats.lognorm.<method>(x, s, scale, loc=0) is evaluated, hence
            'x' must be provided in the prms dictionary (the parameter does not need to
            be called 'x'). Otherwise, the evaluated method is without the primary
            variable hence, scipy.stats.lognorm.<method>(s, scale, loc=0).
        kwargs
            Additional keyword arguments to pass to the specified method.

        Returns
        -------
            The result of scipy.stats.lognorm.<method>(x, s, scale, loc=0) or of
            scipy.stats.lognorm.<method>(s, scale, loc=0).
        """
        mean, std = self.get_hyperparameters(prms, primary_var)
        fun = getattr(stats.lognorm, method)
        # for understanding the following parameter-juggling check out the scipy-docs at
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
        scale = np.exp(mean)  # this is the mean value on the linear scale
        s = std  # the name 's' is used in the scipy-documentation
        if use_primary_var:
            x = prms[primary_var]
            return fun(x, s, scale=scale, **kwargs)
        else:
            return fun(s, scale=scale, **kwargs)

    def plot(
        self,
        primary_var: str,
        ax: "plt.Axes",
        parameters: "Parameters",
        x: Optional[np.ndarray] = None,
        n_points: int = 200,
        color: str = "darkorange",
        rotate: bool = False,
        label="pdf",
    ):
        """
        Plots the prior's probability density function (pdf) to a given axis object.
        For more information on the arguments of this method checkout the parent class's
        :func:`~probeye.definition.distribution.ProbabilityDistribution.plot` method.
        """
        prms = parameters.value_dict
        mean, std = self.get_hyperparameters(prms, primary_var)
        # proceed, only if both values are constants and not latent parameters
        if (mean is not None) and (std is not None):
            if x is None:
                left = 0.0
                right = 2.0 * np.exp(mean)
                x = np.linspace(left, right, n_points)
            else:
                left = min(x)
                right = max(x)
            prms[primary_var] = x
            y = self(primary_var, prms, "pdf")
            margin = 0.025 * abs(right - left)
            adjusted_left = left - margin
            adjusted_right = right + margin
            self._plot(x, y, ax, color, rotate, adjusted_left, adjusted_right, label)


class UniformDistribution(Uniform):
    """
    Uniform probability distribution (univariate, including bounds) with computing
    routines. For more information on the used computing routines check out
    https://en.wikipedia.org/wiki/Continuous_uniform_distribution.
    """

    def __init__(self, prm_dict: dict):
        super().__init__(**prm_dict)

    @staticmethod
    def get_hyperparameters(prms: dict, primary_var: str) -> tuple:
        """
        Extracts the hyperparameters of this distribution from 'prms' and returns them.
        For more information on the arguments check out this class's
        :func:`~probeye.inference.distributions.UniformDistribution.__call__` method.
        """
        low = prms[f"low_{primary_var}"]
        high = prms[f"high_{primary_var}"]
        return low, high

    def __call__(
        self,
        primary_var: str,
        prms: dict,
        method: str,
        use_primary_var: bool = True,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Evaluates scipy.stats.uniform.<method>(x, loc, scale) or, if use_ref_prm=False,
        scipy.stats.uniform.<method>(loc, scale). This function is mostly used with
        method='logpdf' during the sampling procedure.

        Parameters
        ----------
        primary_var
            The primary variable of this distribution. This should mean the variable
            the distribution refers to.
        prms
            Contains the distribution's parameter names as keys and their numeric values
            as the dictionary's values.
        method
            The method of scipy.stats.norm to be evaluated (e.g. 'pdf' or 'logpdf').
        use_primary_var
            If True scipy.stats.uniform.<method>(x, loc, scale) is evaluated, hence 'x'
            must be provided in the prms dictionary (the parameter does not need to be
            called 'x'). Otherwise, the evaluated method is without the primary variable
            hence, scipy.stats.uniform.<method>(loc, scale).
        kwargs
            Additional keyword arguments to pass to the specified method.

        Returns
        -------
            The result of scipy.stats.uniform.<method>(x, loc, scale) or of
            scipy.stats.uniform.<method>(loc, scale).
        """
        low, high = self.get_hyperparameters(prms, primary_var)
        fun = getattr(stats.uniform, method)
        if use_primary_var:
            x = prms[primary_var]
            return fun(x, loc=low, scale=high - low, **kwargs)
        else:
            return fun(loc=low, scale=high - low, **kwargs)

    def plot(
        self,
        primary_var: str,
        ax: "plt.Axes",
        parameters: "Parameters",
        x: Optional[np.ndarray] = None,
        n_points: int = 200,
        color: str = "darkorange",
        rotate: bool = False,
        label="pdf",
    ):
        """
        Plots the prior's probability density function (pdf) to a given axis object.
        For more information on the arguments of this method checkout the parent class's
        :func:`~probeye.definition.distribution.ProbabilityDistribution.plot` method.
        """
        prms = parameters.value_dict
        low, high = self.get_hyperparameters(prms, primary_var)
        # proceed, only if both values are constants and not latent parameters
        if (low is not None) and (high is not None):
            if x is None:
                x = np.linspace(low, high, n_points)
            prms[primary_var] = x
            y = self(primary_var, prms, "pdf")
            margin = 0.025 * abs(high - low)
            adjusted_left = low - margin
            adjusted_right = high + margin
            self._plot(x, y, ax, color, rotate, adjusted_left, adjusted_right, label)


class WeibullDistribution(Weibull):
    """
    Two-parameter Weibull distribution (univariate) with computing routines. For more
    information on the used computing routines check out
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html.
    """

    def __init__(self, prm_dict: dict):
        super().__init__(**prm_dict)

    @staticmethod
    def get_hyperparameters(prms: dict, primary_var: str) -> tuple:
        """
        Extracts the hyperparameters of this distribution from 'prms' and returns them.
        For more information on the arguments check out this class's
        :func:`~probeye.inference.distributions.WeibullDistribution.__call__` method.
        """
        scale = prms[f"scale_{primary_var}"]
        shape = prms[f"shape_{primary_var}"]
        return scale, shape

    def __call__(
        self,
        primary_var: str,
        prms: dict,
        method: str,
        use_primary_var: bool = True,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Evaluates scipy.stats.weibull_min.<method>(x, c, scale, loc=0) or, if
        use_ref_prm=False, scipy.stats.weibull_min.<method>(c, scale, loc=0). This
        function is mostly used with method='logpdf' during the sampling procedure.

        Parameters
        ----------
        primary_var
            The primary variable of this distribution. This should mean the variable
            the distribution refers to.
        prms
            Contains the distribution's parameter names as keys and their numeric values
            as the dictionary's values.
        method
            The method of scipy.stats.weibull_min to be evaluated (e.g. 'logpdf').
        use_primary_var
            If True scipy.stats.weibull_min.<method>(x, c, scale, loc=0) is evaluated,
            hence 'x' must be provided in the prms dictionary (the parameter does not
            need to be called 'x'). Otherwise, the evaluated method is without the
            primary variable hence, scipy.stats.weibull_min.<method>(c, scale, loc=0).
        kwargs
            Additional keyword arguments to pass to the specified method.

        Returns
        -------
            The result of scipy.stats.weibull_min.<method>(x, c, scale, loc=0) or of
            scipy.stats.weibull_min.<method>(c, scale, loc=0).
        """
        scale, shape = self.get_hyperparameters(prms, primary_var)
        c = shape  # the name 'c' is used in the scipy documentation
        fun = getattr(stats.weibull_min, method)
        if use_primary_var:
            x = prms[primary_var]
            return fun(x, c, loc=0, scale=scale, **kwargs)
        else:
            return fun(c, loc=0, scale=scale, **kwargs)

    def plot(
        self,
        primary_var: str,
        ax: "plt.Axes",
        parameters: "Parameters",
        x: Optional[np.ndarray] = None,
        n_points: int = 200,
        color: str = "darkorange",
        rotate: bool = False,
        label="pdf",
    ):
        """
        Plots the prior's probability density function (pdf) to a given axis object.
        For more information on the arguments of this method checkout the parent class's
        :func:`~probeye.definition.distribution.ProbabilityDistribution.plot` method.
        """
        prms = parameters.value_dict
        scale, shape = self.get_hyperparameters(prms, primary_var)
        # proceed, only if both values are constants and not latent parameters
        if (scale is not None) and (shape is not None):
            if x is None:
                x = np.linspace(0, 2.0 * scale, n_points)
            prms[primary_var] = x
            y = self(primary_var, prms, "pdf")
            left = 0.0
            right = 2.0 * scale
            margin = 0.025 * abs(right - left)
            adjusted_left = left - margin
            adjusted_right = right + margin
            self._plot(x, y, ax, color, rotate, adjusted_left, adjusted_right, label)


class SampleBasedDistribution(SampleBased):
    """
    Probability distribution defined via a number of samples (univariate) with computing
    routines. For more info on the used computing routines (Gaussian KDE) check out
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html.
    """

    def __init__(self, prm_dict: dict):
        super().__init__(**prm_dict)
        self.kernel = stats.gaussian_kde(prm_dict["samples"])
        self.mean = np.mean(prm_dict["samples"])
        self.median = np.median(prm_dict["samples"])

    def __call__(
        self,
        primary_var: str,
        prms: dict,
        method: str,
        use_primary_var: bool = True,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Evaluates scipy.stats.gaussian_kde.<method>(x) or. This function is mostly used
        with method='logpdf' during the sampling procedure.

        Parameters
        ----------
        primary_var
            The primary variable of this distribution. This should mean the variable
            the distribution refers to.
        prms
            Contains the distribution's parameter names as keys and their numeric values
            as the dictionary's values.
        method
            The method of scipy.stats.gaussian_kde to be evaluated (e.g. 'logpdf').
        use_primary_var
            Irrelevant for this specific function.
        kwargs
            Additional keyword arguments to pass to the specified method.

        Returns
        -------
            The result of scipy.stats.gaussian_kde.<method>(x).
        """
        # the kernel-class just provides the methods 'pdf' and 'logpdf'; other methods
        # have to be prepared directly
        if method == "mean":
            return self.mean
        elif method == "median":
            return self.median
        elif method == "rvs":
            return np.random.choice(self.prm_dict["samples"], size=kwargs["size"])
        else:
            fun = getattr(self.kernel, method)
            x = prms[primary_var]
            return fun(x, **kwargs)

    def plot(
        self,
        primary_var: str,
        ax: "plt.Axes",
        parameters: "Parameters",
        x: Optional[np.ndarray] = None,
        n_points: int = 200,
        color: str = "darkorange",
        rotate: bool = False,
        label="pdf",
    ):
        """
        Plots the prior's probability density function (pdf) to a given axis object.
        For more information on the arguments of this method checkout the parent class's
        :func:`~probeye.definition.distribution.ProbabilityDistribution.plot` method.
        """
        left = min(self.prm_dict["samples"])
        right = max(self.prm_dict["samples"])
        if x is None:
            x = np.linspace(left, right, n_points)
        prms = parameters.value_dict
        prms[primary_var] = x
        y = self(primary_var, prms, "pdf")
        margin = 0.025 * abs(right - left)
        adjusted_left = left - margin
        adjusted_right = right + margin
        self._plot(x, y, ax, color, rotate, adjusted_left, adjusted_right, label)


def translate_probability_distribution(
    dist_template: ProbabilityDistribution,
) -> ProbabilityDistribution:
    """
    Translate a given instance of ProbabilityDistribution (which is essentially just a
    description of the distribution without compute-methods) to a specific distribution
    object which does contain compute-methods (e.g. compute the logpdf).

    Parameters
    ----------
    dist_template
        An instance of ProbabilityDistribution which contains basic information on the
        probability distribution but no computing-methods.

    Returns
    -------
        An instance of a specific distribution class with computing capabilities.
        Examples for such classes are given above in this file.
    """

    # no translation is required for an uninformative distribution
    if dist_template.dist_type == "uninformative":
        return dist_template

    # this dictionary defines the mapping
    dist_classes = {
        "normal": NormalDistribution,
        "multivariate-normal": MultivariateNormalDistribution,
        "trunc-normal": TruncNormalDistribution,
        "log-normal": LogNormalDistribution,
        "weibull": WeibullDistribution,
        "sample-based": SampleBasedDistribution,
        "uniform": UniformDistribution,
    }

    return dist_classes[dist_template.dist_type](dist_template.prm_dict)
