# standard library
from typing import Union, List, Optional, TYPE_CHECKING

# third party imports
import numpy as np
import matplotlib

# local imports
from probeye.subroutines import translate_prms_def
if TYPE_CHECKING:
    from probeye.definition.parameter import Parameters


class PriorBase:
    """
    Template class for prior definitions. Note that the main motivation of how this
    class is implemented was to avoid storing any numeric values for any of the priors
    parameters within the prior object.
    """

    def __init__(
        self,
        ref_prm: str,
        prms_def: Union[str, dict, List[Union[str, dict]]],
        name: str,
        prior_type: str,
    ):
        """
        Parameters
        ----------
        ref_prm
            The name of the latent parameter the prior refers to.
        prms_def
            A list of strings, or list of one-element-dicts defining the prior's
            parameter names. E.g. ['loc_a', 'scale_a'] or {'loc_a': 'loc_a', 'scale_a':
            'scale_a'}. The latter example is the notation for the use of global and
            local names, which should not be necessary for the definition of prior-
            parameters. A special case is the uninformative prior (see below) which
            hasn't got an parameters. So, in this case prms_def might also be None.
        name
            Defining the priors name.
        prior_type
            Stating the prior's type, e.g. "normal distribution". This is just used for
            printing information on the prior.
        """
        # write arguments to attributes
        self.ref_prm = ref_prm
        self.name = name
        self.prior_type = prior_type

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

    def plot(
        self,
        ax: matplotlib.axes,
        prms: 'Parameters',
        x: Optional[np.ndarray] = None,
        n_points: int = 200,
        n_sigma: Union[int, float] = 2,
        color: str = "darkorange",
    ):
        """
        Plots the prior-pdf to a given axis object.

        Parameters
        ----------
        ax
            The axis object to plot the prior-pdf on.
        prms
            The parameters of the problem at hand.
        x
            The points where the prior-pdf should be evaluated at. If None is
            given, x will be derived from the x-limits of the given ax-object.
        n_points
            The number of points of the prior-pdf graph. Only effective when
            x is None.
        n_sigma
            Defines the x-range of a normal prior via mean plus/minus n_sigma
            times the standard deviation.
        color
            The line-color of the prior-pdf's graph.
        """

        if self.prior_type == "normal":
            mu = prms[self.prms_def[f"loc_{self.ref_prm}"]].value
            sigma = prms[self.prms_def[f"scale_{self.ref_prm}"]].value
            # proceed, only if both values are constants and not latent
            # parameters themselves
            if mu and sigma:
                if x is None:
                    x = np.linspace(
                        mu - n_sigma * sigma, mu + n_sigma * sigma, n_points
                    )
                y = 1 / (sigma * np.sqrt(2 * np.pi))
                y *= np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                ax.plot(x, y, label="prior", color=color)

        elif self.prior_type == "uniform":
            a = prms[self.prms_def[f"low_{self.ref_prm}"]].value
            b = prms[self.prms_def[f"high_{self.ref_prm}"]].value
            # proceed, only if both values are constants and not latent
            # parameters themselves
            if a and b:
                y = np.zeros(n_points)
                y[1:-1] = np.ones(n_points - 2) / (b - a)
                if x is None:
                    x = np.linspace(a, b, n_points)
                    y[0], y[-1] = 0, 0
                else:
                    y[0] = 0 if (x[0] <= a) else y[1]
                    y[-1] = 0 if (x[-1] >= b) else y[-2]
                ax.plot(x, y, label="prior", color=color)
