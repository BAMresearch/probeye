# standard library
import math

# third party imports
import numpy as np

# local imports
from probeye.definition.noise_model import NormalNoiseModel


class NormalNoise(NormalNoiseModel):
    """
    A general Gaussian (normal) noise model with or without correlations. Note
    that currently, there are no correlation capabilities defined.
    """
    def __init__(self, prms_def, sensors, name=None, correlation=None,
                 noise_type='additive'):
        """
        For information on the above arguments check out the docstring of the
        parent class's __init__ method.
        """
        super().__init__(prms_def, sensors, name=name, correlation=correlation,
                         noise_type=noise_type)

    def loglike_contribution(self, model_response, prms):
        """
        This method overwrites the corresponding method of the parent class.
        Check out the docstring there for additional information.
        """
        # compute the model error; note that this mode has exactly one sensor
        model_error_vector = self.error(model_response)[self.sensors[0]]
        # the precision 'prec' is defined as the inverse of the variance, hence
        # prec = 1 / sigma**2 where sigma denotes the standard deviation
        std = prms['std']
        mean = 0.0 if self.zero_mean else prms['mean']
        prec = 1.0 / std ** 2.0
        ll = 0.0
        # evaluate the Gaussian log-PDF with zero mean and a variance of
        # 1/prec for each error term and sum them up
        ll -= len(model_error_vector) / 2 * math.log(2 * math.pi / prec)
        ll -= 0.5 * prec * np.sum(np.square(model_error_vector - mean))
        return ll

def translate_noise_model(noise_base):
    """
    Translates a given instance of NoiseBase (which is essentially just a
    description of the noise model without computing-methods) to a specific
    noise model object which does contain computing-methods (e.g. compute the
    log-likelihood contribution).

    Parameters
    ----------
    noise_base : obj[NoiseBase]
        An instance of NoiseBase which contains basic information on the noise
        model but no computing-methods.

    Returns
    -------
    noise_object : obj[NoiseBase]
        An instance of a specific noise model class with computing capabilities.
        Examples for such classes are given above in this file.
    """

    # this is the noise classes currently defined (see code above)
    noise_classes = {'normal': NormalNoise}

    # this is where the translation happens
    noise_object = noise_classes[noise_base.dist](
        prms_def=noise_base.prms_def, sensors=noise_base.sensors,
        name=noise_base.name, correlation=noise_base.correlation,
        noise_type=noise_base.noise_type)

    # here, we take the assigned experiments from the base object
    noise_object.experiment_names = noise_base.experiment_names
    noise_object.problem_experiments = noise_base.problem_experiments

    return noise_object
