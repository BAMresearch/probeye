# standard library
from typing import Union, List, Optional, Tuple

# third party imports
import numpy as np

# local imports
from probeye.definition.inverse_problem import InverseProblem


class BiasedInverseProblem(InverseProblem):
    """
    This class provides a framework for defining an biased inverse problem. It simply extends
    the InverseProblem class by adding a bias model as a Gaussian Process specified by a set of
    parameters.

    Parameters
    ----------
    name
        This is the name of the problem and has only descriptive value, for example when
        working with several inverse problems.
    use_default_logger
        When True, the logger will be set up with some useful default values. Otherwise,
        no logger configurations are applied and a logger can be defined outside of the
        problem definition.
    log_level
        The log-level used by the default logger for printing to std out. This argument
        is intended for quickly controlling the amount of logging output the user sees
        on the screen when running probeye.
    log_file
        Path to the log-file, if the logging-stream should be printed to file. If None
        is given, no logging-file will be created.
    print_header
        If True, a probeye header is logged when an instance of this class is created.
        Otherwise, the header will not be logged.
    bias_parameters
        A dictionary containing the parameters of the bias model. The keys depend on the
        type of bias model used.
    """

    def add_bias_model(self, bias_model_class, bias_parameters: dict):
        """
        Adds a bias model to the inverse problem.

        Parameters
        ----------
        bias_model
            An instance of a bias model.
        """

        # Check if the bias model has a train method
        if not hasattr(bias_model_class, "train"):
            raise Exception("The bias model must have a train method.")
        self.bias_model_class = bias_model_class
        self.bias_parameters = bias_parameters
        # self.bias_model = bias_model_class(**bias_parameters)

