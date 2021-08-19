# standard library imports
import math

# third party imports
import numpy as np

# local imports
from probeye.subroutines import len_or_one
from probeye.subroutines import list2dict


class NoiseTemplate:
    def __init__(self, prms_def):
        """
        Parameters
        ----------
        prms_def : list
            A list of parameter names (strings) defining how a noise parameter
            vector given to the loglike_contribution method is interpreted. E.g.
            prms_def = ['mu', 'sigma'] means that the noise parameter vector
            has two elements, the first of which gives the value of 'mu' and the
            second gives the value of 'sigma'.
        """
        self.prms_def = list2dict(prms_def)
        self.prms_dim = len_or_one(prms_def)

    def loglike_contribution(self, model_error_vector, prms):
        """
        Evaluates the log-likelihood function for the given model error and
        the given noise parameter vector. This method has to be overwritten.

        Parameters
        ----------
        model_error_vector : array_like
            Usually a 1D numerical array containing the model errors.
        prms : ParameterList-object
            Dictionary-like object containing parameter name:value pairs.

        Returns
        -------
        ll : float
            The evaluated log-likelihood function.
        """
        raise NotImplementedError(
            "Your model does not have a loglike_contribution-method. You " +
            "need to define this method so you can evaluate your noise model."
        )


class NormalNoiseZeroMean(NoiseTemplate):
    """
    A simple Gaussian (normal) zero-mean noise model without any correlations.
    """
    def __init__(self, prms_def):
        super().__init__(prms_def)
        if len(self.prms_def) > 1:
            raise RuntimeError(
                f"NormalNoise allows to define exactly one parameter (the"
                f"standard deviation), which must be given as a list with one"
                f"element. However, you provided: '{prms_def}'"
            )
        # the one parameter name that is given refers to the std. deviation
        self.sigma_name = list(self.prms_def.keys())[0]

    def loglike_contribution(self, model_error_vector, prms):
        """
        This method overwrites the corresponding method of the parent class.
        Check out the docstring there for additional information.
        """
        # the precision 'prec' is defined as the inverse of the variance, hence
        # prec = 1 / sigma**2 where sigma denotes the standard deviation
        sigma = prms[self.sigma_name]
        prec = 1.0 / sigma**2-0
        ll = 0.0
        # evaluate the Gaussian log-PDF with zero mean and a variance of
        # 1/prec for each error term and sum them up
        ll -= len(model_error_vector) / 2 * math.log(2 * math.pi / prec)
        ll -= 0.5 * prec * np.sum(np.square(model_error_vector))
        return ll
