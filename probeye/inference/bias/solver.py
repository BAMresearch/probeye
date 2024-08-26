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
import chaospy #FIXME: This should not be always imported

# local imports
from probeye.inference.emcee.solver import EmceeSolver
from probeye.inference.bias.likelihood_models import translate_likelihood_model
from probeye.subroutines import vectorize_nd_numpy_dict, vectorize_numpy_dict, vectorize_tuple_pce_dict
from probeye.subroutines import pretty_time_delta
from probeye.subroutines import stream_to_logger
from probeye.subroutines import print_dict_in_rows

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
            fwd_model = self.problem.forward_models[fwd_name]
            self.problem.likelihood_models[like_name].forward_model = fwd_model
            self.problem.likelihood_models[like_name].determine_output_lengths()

            # translate the likelihood model
            self.problem.likelihood_models[like_name] = translate_likelihood_model(
                self.problem.likelihood_models[like_name]
            )

    def evaluate_model_response(
        self, theta: np.ndarray, forward_model: "ForwardModelBase", experiment_name: str
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
            fwd_model = self.problem.forward_models[fwd_name]
            self.problem.likelihood_models[like_name].forward_model = fwd_model
            self.problem.likelihood_models[like_name].determine_output_lengths()

            # translate the likelihood model
            self.problem.likelihood_models[like_name] = translate_likelihood_model(
                self.problem.likelihood_models[like_name]
            )

    def evaluate_model_response(
        self, theta: np.ndarray, forward_model: "ForwardModelBase", experiment_name: str
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
        inp["experiment_name"] = experiment_name # Required for surrogate selection
        
        # evaluate the forward model and translate the result to a single vector
        model_response_dict = forward_model(inp)
        model_response_vector, dist = vectorize_tuple_pce_dict(model_response_dict)
        mean_response_vector = np.array([chaospy.E(response, dist) for response in model_response_vector]).flatten()
        std_response_vector = np.array([chaospy.Std(response, dist) for response in model_response_vector]).flatten()
        model_response_vector = np.array([mean_response_vector, std_response_vector])

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
    
    def run(
        self,
        n_walkers: int = 20,
        n_steps: int = 1000,
        n_initial_steps: int = 100,
        true_values: Optional[dict] = None,
        **kwargs,
    ) -> az.data.inference_data.InferenceData:
        """
        Runs the emcee-sampler for the InverseProblem the EmceeSolver was initialized
        with and returns the results as an arviz InferenceData obj.

        Parameters
        ----------
        n_walkers
            Number of walkers used by the estimator.
        n_steps
            Number of steps to run.
        n_initial_steps
            Number of steps for initial (burn-in) sampling.
        true_values
            True parameter values, if known.
        kwargs
            Additional key-word arguments channeled to emcee.EnsembleSampler.

        Returns
        -------
        inference_data
            Contains the results of the sampling procedure.
        """

        # log which solver is used
        logger.info(
            f"Solving problem using emcee sampler with {n_initial_steps} + {n_steps} "
            f"samples and {n_walkers} walkers"
        )
        if kwargs:
            logger.info("Additional options:")
            print_dict_in_rows(kwargs, printer=logger.info)
        else:
            logger.info("No additional options specified")

        # draw initial samples from the parameter's priors
        logger.debug("Drawing initial samples")
        if self.seed is not None:
            np.random.seed(self.seed)
        sampling_initial_positions = np.zeros(
            (n_walkers, self.problem.n_latent_prms_dim)
        )
        theta_names = self.problem.get_theta_names(tex=False, components=False)
        for parameter_name in theta_names:
            idx = self.problem.parameters[parameter_name].index
            idx_end = self.problem.parameters[parameter_name].index_end
            samples = self.sample_from_prior(parameter_name, n_walkers)
            if (idx_end - idx) == 1:
                sampling_initial_positions[:, idx] = samples
            else:
                sampling_initial_positions[:, idx:idx_end] = samples

        # The following code is based on taralli and merely adjusted to the variables
        # in the probeye setup; see https://gitlab.com/tno-bim/taralli

        # ............................................................................ #
        #                                 Pre-process                                  #
        # ............................................................................ #

        def logprob(x):
            # Skip loglikelihood evaluation if logprior is equal
            # to negative infinity
            logprior = self.logprior(x)
            if logprior == -np.inf:
                return logprior

            # Otherwise return logprior + loglikelihood
            return logprior + self.loglike(x)

        logger.debug("Setting up EnsembleSampler")
        self.sampler = emcee.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=self.problem.n_latent_prms_dim,
            log_prob_fn=logprob,
            **kwargs,
        )

        if self.seed is not None:
            random.seed(self.seed)
            self.sampler.random_state = np.random.mtrand.RandomState(self.seed)

        # ............................................................................ #
        #        Initial sampling, burn-in: used to avoid a poor starting point        #
        # ............................................................................ #

        logger.debug("Starting sampling (initial + main)")
        start = time.time()
        state = self.sampler.run_mcmc(
            initial_state=sampling_initial_positions,
            nsteps=n_initial_steps,
            progress=self.show_progress,
        )
        self.sampler.reset()

        # ............................................................................ #
        #                          Sampling of the posterior                           #
        # ............................................................................ #
        self.sampler.run_mcmc(
            initial_state=state, nsteps=n_steps, progress=self.show_progress
        )
        end = time.time()
        runtime_str = pretty_time_delta(end - start)
        logger.info(
            f"Sampling of the posterior distribution completed: {n_steps} steps and "
            f"{n_walkers} walkers."
        )
        logger.info(f"Total run-time (including initial sampling): {runtime_str}.")
        logger.info("")
        logger.info("Summary of sampling results (emcee)")
        posterior_samples = self.sampler.get_chain(flat=True)
        with contextlib.redirect_stdout(stream_to_logger("INFO")):  # type: ignore
            self.summary = self.emcee_summary(
                posterior_samples, true_values=true_values
            )
        logger.info("")  # empty line for visual buffer
        self.raw_results = self.sampler

        # translate the results to a common data structure and return it
        self.var_names = self.problem.get_theta_names(tex=True, components=True)
        inference_data = az.from_emcee(self.sampler, var_names=self.var_names)
        return inference_data

    def restart_run(self, state, n_steps):
        """
        Restart the emcee-sampler for the InverseProblem the EmceeSolver was initialized
        with and returns the results as an arviz InferenceData obj.

        Parameters
        ----------
        state
            The state of the sampler to restart from.
        n_steps
            Number of steps to run.
        """

        self.sampler.run_mcmc(
            initial_state=state, nsteps=n_steps, progress=False)