# standard library imports
import math

# third party imports
import numpy as np

# local imports
from probeye.definition.noise_model import NoiseModelTemplate


class NormalNoise(NoiseModelTemplate):
    """
    A simple Gaussian (normal) zero-mean noise model without any correlations.
    """
    def __init__(self, prms_def, sensors, noise_type='additive'):
        super().__init__(prms_def, sensors)
        if self.prms_dim > 1:
            raise RuntimeError(
                f"NormalNoise allows to define exactly one parameter (the"
                f"standard deviation), which must be given as a list with one"
                f"element. However, you provided: '{prms_def}'")
        if len(self.sensors) > 1:
            raise RuntimeError(
                f"This noise-model refers to exactly one sensor. "
                f"However, {len(self.sensors)} were given.")
        # the one parameter name that is given refers to the std. deviation
        self.sigma_name = list(self.prms_def.keys())[0]
        self.noise_type = noise_type

        # set the error_function depending on the noise-type
        if noise_type == "additive":
            self.error_function = self.error_function_additive
        elif noise_type == "multiplicative":
            self.error_function = self.error_function_multiplicative
        else:
            raise ValueError(
                f"Encountered unknown noise_type: '{noise_type}'. The noise"
                f"_type must be either 'additive' or 'multiplicative'."
            )

    def error_function(self, ym_dict, ye_dict):
        """
        This function will be defined as either error_function_additive or
        error_function_multiplicative during the __init__-call of this class.
        Check out one of these methods for more information on the input args.
        """

    def error_function_additive(self, ym_dict, ye_dict):
        """
        Evaluates the additive model error for each of the noise model' sensors.

        Parameters
        ----------
        ym_dict : dict
            The computed values for the model's output sensor_values.
        ye_dict : dict
            The measured values for the model's output sensor_values.

        Returns
        -------
        error_dict : dict
            The computed model error for the model's output sensor_values.
        """
        # for each sensor, its own error metric is used to compute the error
        error_dict = {name: ym_dict[name] - ye_dict[name]
                      for name in self.sensors}
        return error_dict

    def error_function_multiplicative(self, ym_dict, ye_dict):
        """
        Evaluates the multiplicative model error for each of the noise model's
        sensors.

        Parameters
        ----------
        ym_dict : dict
            The computed values for the model's output sensor_values.
        ye_dict : dict
            The measured values for the model's output sensor_values.

        Returns
        -------
        error_dict : dict
            The computed model error for the model's output sensor_values.
        """
        # for each sensor, its own error metric is used to compute the error
        error_dict = {name: ym_dict[name] / ye_dict[name] - 1.0
                      for name in self.sensors}
        return error_dict

    def loglike_contribution(self, model_response, prms, problem_experiments):
        """
        This method overwrites the corresponding method of the parent class.
        Check out the docstring there for additional information.
        """
        # compute the model error; note that this mode has exactly one sensor
        model_error_vector = self.error(
            model_response, problem_experiments)[self.sensors[0]]
        # the precision 'prec' is defined as the inverse of the variance, hence
        # prec = 1 / sigma**2 where sigma denotes the standard deviation
        sigma = prms[self.sigma_name]
        prec = 1.0 / sigma**2.0
        ll = 0.0
        # evaluate the Gaussian log-PDF with zero mean and a variance of
        # 1/prec for each error term and sum them up
        ll -= len(model_error_vector) / 2 * math.log(2 * math.pi / prec)
        ll -= 0.5 * prec * np.sum(np.square(model_error_vector))
        return ll
