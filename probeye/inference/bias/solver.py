# standard library imports
from typing import TYPE_CHECKING, Optional, Union, Callable, Tuple

# third party imports
import numpy as np
from loguru import logger
import emcee
import arviz as az
import time
import random
import contextlib
import chaospy  # FIXME: This should not be always imported

# local imports
from probeye.definition.forward_model import ForwardModelBase
from probeye.inference.emcee.solver import EmceeSolver
from probeye.subroutines import (
    vectorize_nd_numpy_dict,
    vectorize_numpy_dict,
    vectorize_tuple_pce_dict,
)
from probeye.subroutines import pretty_time_delta
from probeye.subroutines import stream_to_logger
from probeye.subroutines import print_dict_in_rows

from multiprocessing import Pool  # pickling problem

# from multiprocessing.pool import ThreadPool as Pool # no pickling needed but no time effect
import os

if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.inverse_problem import InverseProblem

os.environ["OMP_NUM_THREADS"] = "1"
logprob = None


class EmbeddedMCISolver(EmceeSolver):
    """
    Solver for the embedded method with Monte-Carlo Integration.
    """

    def _translate_likelihood_models(self):
        """
        Translate the inverse problem's likelihood models as needed for this solver.
        """

        logger.debug("Translate the problem's likelihood models")
        for like_name in self.problem.likelihood_models:
            # the likelihood model's forward model is still referencing the old (i.e.,
            # not-translated) forward model and needs to be reset to the updated one
            fwd_name = self.problem.likelihood_models[like_name].forward_model.name
            fwd_model = self.problem.internal_forward_models[fwd_name]
            self.problem.likelihood_models[like_name].forward_model = fwd_model
            self.problem.likelihood_models[like_name].determine_output_lengths()

    def evaluate_model_response(
        self,
        theta: np.ndarray,
        forward_model: ForwardModelBase,
        experiment_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the model response for each forward model for the given parameter
        vector theta and the given experiments.

        Parameters
        ----------
        theta
            A numeric vector for which the model responses should be evaluated. Which
            parameters these numbers refer to can be checked by calling
            self.theta_explanation() once the problem is set up.
        forward_model
            The forward model that should be evaluated.
        experiment_name
            The experiment, the forward model should be evaluated for.

        Returns
        -------
        model_response_vector
            Vector of the model responses (concatenated over output sensors).
        residuals_vector
            Vector of the model residuals (concatenated over output sensors).
        """

        # prepare the input dictionary for the forward model call
        prms_model = self.problem.get_parameters(theta, forward_model.prms_def)
        exp_inp = forward_model.input_from_experiments[experiment_name]
        inp = {**exp_inp, **prms_model}  # adds the two dictionaries

        # evaluate the forward model and translate the result to a single vector
        model_response_dict = forward_model(inp)
        model_response_vector = vectorize_nd_numpy_dict(model_response_dict)
        mean_response_vector = np.mean(model_response_vector, axis=0)
        var_response_vector = np.var(model_response_vector, axis=0)
        model_response_vector = np.array([mean_response_vector, var_response_vector])

        # compute the residuals by comparing to the experimental response
        exp_response_dict = forward_model.output_from_experiments[experiment_name]
        # Reorder exmperiment response dict to match model response dict
        if not list(model_response_dict.keys()) == list(exp_response_dict.keys()):
            exp_response_dict = {
                key: exp_response_dict[key] for key in model_response_dict.keys()
            }
            forward_model.output_from_experiments[experiment_name] = exp_response_dict
        exp_response_vector = vectorize_numpy_dict(exp_response_dict)
        residuals_vector = exp_response_vector - mean_response_vector

        return model_response_vector, residuals_vector


class EmbeddedPCESolver(EmceeSolver):
    """
    Solver for the embedded method with Polynomial Chaos Expansion.
    """

    def _translate_likelihood_models(self):
        """
        Translate the inverse problem's likelihood models as needed for this solver.
        """

        logger.debug("Translate the problem's likelihood models")
        for like_name in self.problem.likelihood_models:
            # the likelihood model's forward model is still referencing the old (i.e.,
            # not-translated) forward model and needs to be reset to the updated one
            fwd_name = self.problem.likelihood_models[like_name].forward_model.name
            fwd_model = self.problem.internal_forward_models[fwd_name]
            self.problem.likelihood_models[like_name].forward_model = fwd_model
            self.problem.likelihood_models[like_name].determine_output_lengths()

    def evaluate_model_response(
        self,
        theta: np.ndarray,
        forward_model: ForwardModelBase,
        experiment_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the model response for each forward model for the given parameter
        vector theta and the given experiments.

        Parameters
        ----------
        theta
            A numeric vector for which the model responses should be evaluated. Which
            parameters these numbers refer to can be checked by calling
            self.theta_explanation() once the problem is set up.
        forward_model
            The forward model that should be evaluated.
        experiment_name
            The experiment, the forward model should be evaluated for.

        Returns
        -------
        model_response_vector
            Vector of the model responses (concatenated over output sensors).
        residuals_vector
            Vector of the model residuals (concatenated over output sensors).
        """

        # prepare the input dictionary for the forward model call
        prms_model = self.problem.get_parameters(theta, forward_model.prms_def)
        exp_inp = forward_model.input_from_experiments[experiment_name]
        inp = {**exp_inp, **prms_model}  # adds the two dictionaries
        inp["experiment_name"] = experiment_name  # Required for surrogate selection

        # evaluate the forward model and translate the result to a single vector
        model_response_dict = forward_model(inp)
        model_response_vector, dist = vectorize_tuple_pce_dict(model_response_dict)
        mean_response_vector = np.array(
            [chaospy.E(response, dist) for response in model_response_vector]
        ).flatten()
        std_response_vector = np.array(
            [chaospy.Std(response, dist) for response in model_response_vector]
        ).flatten()
        model_response_vector = np.array([mean_response_vector, std_response_vector])

        # compute the residuals by comparing to the experimental response
        exp_response_dict = forward_model.output_from_experiments[experiment_name]
        # Reorder experiment response dict to match model response dict
        if not list(model_response_dict.keys()) == list(exp_response_dict.keys()):
            exp_response_dict = {
                key: exp_response_dict[key] for key in model_response_dict.keys()
            }
            forward_model.output_from_experiments[experiment_name] = exp_response_dict
        exp_response_vector = vectorize_numpy_dict(exp_response_dict)
        residuals_vector = exp_response_vector - mean_response_vector

        return model_response_vector, residuals_vector