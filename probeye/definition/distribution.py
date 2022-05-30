# standard library
from typing import Union, Optional, TYPE_CHECKING

# third party imports
import numpy as np

# this is required to prevent a circular import-structure
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.parameter import Parameters
    import matplotlib.pyplot as plt


class ProbabilityDistribution:
    """
    Base class for the different (specific) probability distributions defined below.
    All of these classes have in common that they merely describe the respective
    distribution without providing any computing routines. The latter are overloaded
    when the problem is handed over to a solver.

    Parameters
    ----------
    dist_type
        The type of the distribution. For example 'normal' or 'uniform'.
    """

    def __init__(self, dist_type: str):

        # this attribute will be defined by the more specific classes given below
        self.dist_type = dist_type

        # this attribute will contain the distribution's hyperparameters as keys and the
        # given values (either numeric or string) as the dictionary's values
        self.prm_dict = {}  # type: dict

    def __call__(
        self,
        primary_var: str,
        prms: dict,
        method: str,
        use_primary_var: bool = True,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        """
        Template for evaluating the probability distribution. For example, to evaluate
        the probability density function.

        Parameters
        ----------
        primary_var
            The primary variable of this distribution. This should mean the variable
            the distribution refers to.
        prms
            Contains the distribution's parameter names as keys and their numeric values
            as the dictionary's values.
        method
            The method of to be evaluated (e.g. 'pdf' or 'logpdf').
        use_primary_var
            Defines whether or not the primary variable must be passed for the requested
            evaluation.
        kwargs
            Additional keyword arguments to pass to the specified method.

        Returns
        -------
            The result of the evaluation.
        """
        raise NotImplementedError("The '__call__'-method has not been implemented yet!")

    @staticmethod
    def _plot(
        x: Union[np.ndarray, list, tuple, float, int],
        y: Union[np.ndarray, list, tuple, float, int],
        ax: "plt.Axes",
        color: str,
        rotate: bool,
        adjusted_left: Union[int, float],
        adjusted_right: Union[int, float],
        label: str,
    ):
        """
        Basic plotting function for plotting the distributions probability density
        function (pdf).

        Parameters
        ----------
        x
            The values on the (un-rotated) x-axis of the pdf-plot.
        y
            The values on the (un-rotated) y-axis of the pdf-plot.
        ax
            The axis object to plot the prior-pdf on.
        color
            The line-color of the prior-pdf's graph.
        rotate
            If True, the x- and y-axis are switched. This is required, for example, in
            a pair-plot with two parameters. Here, the histogram to the right is rotated
            by 90 degrees.
        adjusted_left
            Left limit of (un-rotated) x-axis. Typically adjusted with a margin.
        adjusted_right
            Right limit of (un-rotated) x-axis. Typically adjusted with a margin.
        label
            The label used in the legend of the plot for the plotted pdf.
        """
        if not rotate:  # default case
            ax.plot(x, y, label=label, color=color)
            ax.set_xlim(adjusted_left, adjusted_right)
        else:  # x-axis and y-axis switch roles
            ax.plot(y, x, label=label, color=color)
            ax.set_ylim(adjusted_left, adjusted_right)

    def plot(
        self,
        primary_var: str,
        ax: "plt.Axes",
        prms: "Parameters",
        x: Optional[np.ndarray] = None,
        n_points: int = 200,
        color: str = "darkorange",
        rotate: bool = False,
        label="pdf",
    ):
        """
        Template for plotting method which plots the prior-pdf to a given axis object.

        Parameters
        ----------
        primary_var
            The reference parameter of the distribution. For example, if one considers
            a normal distribution with a density called f, which one would evaluate via
            f(x, mean=0, std=1), then x would be the reference parameter.
        ax
            The axis object to plot the prior-pdf on.
        prms
            The parameters of the problem at hand. Essentially a dictionary. But the
            values are ParameterProperties-objects.
        x
            The points where the prior-pdf should be evaluated at. If None is given, x
            will be derived from the x-limits of the given ax-object.
        n_points
            The number of points of the prior-pdf graph. Only effective when x is None.
        color
            The line-color of the prior-pdf's graph.
        rotate
            If True, the x- and y-axis are switched. This is required, for example, in
            a pair-plot with two parameters. Here, the histogram to the right is rotated
            by 90 degrees.
        label
            The label used in the legend of the plot for the plotted pdf.
        """
        raise NotImplementedError("The 'plot'-method has not been implemented yet!")


class Normal(ProbabilityDistribution):
    """
    Normal or Gaussian probability distribution (univariate). For more information
    check out https://en.wikipedia.org/wiki/Normal_distribution.

    Parameters
    ----------
    mean
        The mean value of the distribution. Either a number, or a string that describes
        the name of the parameter that defines the value.
    std
        The standard deviation of the distribution. Either a number, or a string that
        describes the name of the parameter that defines the value.
    """

    def __init__(self, mean: Union[int, float, str], std: Union[int, float, str]):
        super().__init__(dist_type="normal")
        self.prm_dict = {"mean": mean, "std": std}


class MultivariateNormal(ProbabilityDistribution):
    """
    Normal or Gaussian probability distribution (multivariate). For more information
    check out https://en.wikipedia.org/wiki/Multivariate_normal_distribution.

    Parameters
    ----------
    mean
        The mean value of the distribution. Either a vector, or a string that describes
        the name of the parameter that defines the value.
    cov
        The covariance matrix of the distribution. Either an array, or a string that
        describes the name of the parameter that defines the value.
    """

    def __init__(self, mean: Union[np.ndarray, str], cov: Union[np.ndarray, str]):
        super().__init__(dist_type="multivariate-normal")
        self.prm_dict = {"mean": mean, "cov": cov}


class TruncNormal(ProbabilityDistribution):
    """
    Truncated normal or Gaussian probability distribution (univariate). For more
    information check out https://en.wikipedia.org/wiki/Truncated_normal_distribution.

    Parameters
    ----------
    mean
        The mean value of the distribution. Either a number, or a string that describes
        the name of the parameter that defines the value.
    std
        The standard deviation of the distribution. Either a number, or a string that
        describes the name of the parameter that defines the value.
    low
        The lower bound of the distribution. Either a number, or a string that describes
        the name of the parameter that defines the value.
    high
        The upper bound of the distribution. Either a number, or a string that describes
        the name of the parameter that defines the value.
    """

    def __init__(
        self,
        mean: Union[int, float, str],
        std: Union[int, float, str],
        low: Union[int, float, str],
        high: Union[int, float, str],
    ):
        super().__init__(dist_type="trunc-normal")
        self.prm_dict = {"mean": mean, "std": std, "low": low, "high": high}


class LogNormal(ProbabilityDistribution):
    """
    Log-normal probability distribution (univariate). For more information check out
    https://en.wikipedia.org/wiki/Log-normal_distribution.

    Parameters
    ----------
    mean
        The mean value of the distribution on the log-scale. Either a number, or a
        string that describes the name of the parameter that defines the value.
    std
        The standard deviation of the distribution on the log-scale. Either a number, or
        a string that describes the name of the parameter that defines the value.
    """

    def __init__(self, mean: Union[int, float, str], std: Union[int, float, str]):
        super().__init__(dist_type="log-normal")
        self.prm_dict = {"mean": mean, "std": std}


class Uniform(ProbabilityDistribution):
    """
    Uniform probability distribution (univariate) with bounds included. For more
    information check out https://en.wikipedia.org/wiki/Continuous_uniform_distribution.

    Parameters
    ----------
    low
        The lower bound of the distribution. Either a number, or a string that describes
        the name of the parameter that defines the value.
    high
        The upper bound of the distribution. Either a number, or a string that describes
        the name of the parameter that defines the value.
    """

    def __init__(self, low: Union[int, float, str], high: Union[int, float, str]):
        super().__init__(dist_type="uniform")
        self.prm_dict = {"low": low, "high": high}


class Weibull(ProbabilityDistribution):
    """
    Two-parameter Weibull distribution (univariate). For more information check out
    https://en.wikipedia.org/wiki/Weibull_distribution.

    Parameters
    ----------
    scale
        The scale-parameter of the distribution. Either a number, or a string that
        describes the name of the parameter that defines the value.
    shape
        The shape-parameter of the distribution. Either a number, or a string that
        describes the name of the parameter that defines the value.
    """

    def __init__(self, scale: Union[int, float, str], shape: Union[int, float, str]):
        super().__init__(dist_type="weibull")
        self.prm_dict = {"scale": scale, "shape": shape}


class SampleBased(ProbabilityDistribution):
    """
    Probability distribution defined via a number of samples (univariate). For more
    information check out https://en.wikipedia.org/wiki/Sampling_distribution.

    Parameters
    ----------
    samples
        The samples, the distribution is based on. Either a vector, or a string that
        describes the name of the parameter that defines the value.
    """

    def __init__(self, samples: Union[np.ndarray, str]):
        super().__init__(dist_type="sample-based")
        self.prm_dict = {"samples": samples}


class Uninformative(ProbabilityDistribution):
    """
    Represents a univariate uniform distribution with bounds "at" +/- infinity. This
    distribution (whose internal purpose is that of a tag) is automatically assigned to
    a parameter that is added to an inverse problem that is neither assigned a prior nor
    a value. For example, problem.add_parameter("a").
    """

    def __init__(self):
        super().__init__(dist_type="uninformative")
