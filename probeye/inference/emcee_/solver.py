# standard library imports
from typing import TYPE_CHECKING, Optional
import time
import random
import contextlib

# third party imports
import numpy as np
import emcee
import arviz as az
from loguru import logger
from tabulate import tabulate

# local imports
from probeye.subroutines import pretty_time_delta
from probeye.subroutines import check_for_uninformative_priors
from probeye.inference.scipy_.solver import ScipySolver
from probeye.subroutines import stream_to_logger
from probeye.subroutines import print_dict_in_rows
from probeye.subroutines import extract_true_values

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.inference_problem import InferenceProblem


class EmceeSolver(ScipySolver):
    """
    Provides emcee-sampler which is a pure-Python implementation of Goodman & Weare’s
    Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler. For more
    information, check out https://emcee.readthedocs.io/en/stable/.
    """

    def __init__(
        self, problem: "InferenceProblem", seed: int = 1, show_progress: bool = True
    ):
        """See docstring of ScipySolver for information on the arguments."""
        logger.debug("Initializing EmceeSolver")
        # check that the problem does not contain a uninformative prior
        check_for_uninformative_priors(problem)
        # initialize the scipy-based solver (ScipySolver)
        super().__init__(problem, seed=seed, show_progress=show_progress)

    def emcee_summary(
        self, posterior_samples: np.ndarray, true_values: Optional[dict] = None
    ) -> dict:
        """
        Computes and prints a summary of the posterior samples containing mean, median,
        standard deviation, 5th percentile and 95th percentile. Note, that this method
        was based on code from the taralli package: https://gitlab.com/tno-bim/taralli.

        Parameters
        ----------
        posterior_samples
            The generated samples in an array with as many columns as there are latent
            parameters, and n rows, where n = n_chains * n_steps.
        true_values
            True parameter values, if known.

        Returns
        -------
            Keys are the different statistics 'mean', 'median', 'sd' (standard
            deviation), 'q05' and 'q95' (0.05- and 0.95-quantile). The values are
            dictionaries with the parameter names as keys and the respective statistics
            as values.
        """

        # used for the names in the first column
        var_names = self.problem.get_theta_names(tex=False, components=True)

        # compute some stats for each column (i.e., each parameter)
        mean = np.mean(posterior_samples, axis=0)
        quantiles = np.quantile(posterior_samples, [0.50, 0.05, 0.95], axis=0)
        median = quantiles[0, :]
        quantile_05 = quantiles[1, :]
        quantile_95 = quantiles[2, :]

        # compute the sample standard deviations for each parameter
        cov_matrix = np.atleast_2d(np.cov(posterior_samples.T))
        sd = np.sqrt(np.diag(cov_matrix))

        # assemble the summary array
        if true_values:
            col_names = ["", "true", "mean", "median", "sd", "5%", "95%"]
            true = extract_true_values(true_values, var_names)
            row_names = np.array(var_names)
            tab = np.hstack(
                (
                    row_names.reshape(-1, 1),
                    true.reshape(-1, 1),
                    mean.reshape(-1, 1),
                    median.reshape(-1, 1),
                    sd.reshape(-1, 1),
                    quantile_05.reshape(-1, 1),
                    quantile_95.reshape(-1, 1),
                )
            )

            # print the generated table, and return a summary dict for later use
            print(tabulate(tab, headers=col_names, floatfmt=".2f"))
            return {
                "true": {name: val for name, val in zip(row_names, true)},
                "mean": {name: val for name, val in zip(row_names, mean)},
                "median": {name: val for name, val in zip(row_names, median)},
                "sd": {name: val for name, val in zip(row_names, sd)},
                "q05": {name: val for name, val in zip(row_names, quantile_05)},
                "q95": {name: val for name, val in zip(row_names, quantile_95)},
            }
        else:
            col_names = ["", "mean", "median", "sd", "5%", "95%"]
            row_names = np.array(var_names)
            tab = np.hstack(
                (
                    row_names.reshape(-1, 1),
                    mean.reshape(-1, 1),
                    median.reshape(-1, 1),
                    sd.reshape(-1, 1),
                    quantile_05.reshape(-1, 1),
                    quantile_95.reshape(-1, 1),
                )
            )

            # print the generated table, and return a summary dict for later use
            print(tabulate(tab, headers=col_names, floatfmt=".2f"))
            return {
                "mean": {name: val for name, val in zip(row_names, mean)},
                "median": {name: val for name, val in zip(row_names, median)},
                "sd": {name: val for name, val in zip(row_names, sd)},
                "q05": {name: val for name, val in zip(row_names, quantile_05)},
                "q95": {name: val for name, val in zip(row_names, quantile_95)},
            }

    def run_mcmc(
        self,
        n_walkers: int = 20,
        n_steps: int = 1000,
        n_initial_steps: int = 100,
        true_values: Optional[dict] = None,
        **kwargs,
    ) -> az.data.inference_data.InferenceData:
        """
        Runs the emcee-sampler for the InferenceProblem the EmceeSolver was initialized
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

        random.seed(self.seed)
        np.random.seed(self.seed)
        rstate = np.random.mtrand.RandomState(self.seed)

        logger.debug("Setting up EnsembleSampler")
        sampler = emcee.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=self.problem.n_latent_prms_dim,
            log_prob_fn=lambda x: self.logprior(x) + self.loglike(x),
            **kwargs,
        )

        sampler.random_state = rstate

        # ............................................................................ #
        #        Initial sampling, burn-in: used to avoid a poor starting point        #
        # ............................................................................ #

        logger.debug("Starting sampling (initial + main)")
        start = time.time()
        state = sampler.run_mcmc(
            initial_state=sampling_initial_positions,
            nsteps=n_initial_steps,
            progress=self.show_progress,
        )
        sampler.reset()

        # ............................................................................ #
        #                          Sampling of the posterior                           #
        # ............................................................................ #
        sampler.run_mcmc(
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
        logger.info("Summary of sampling results")
        posterior_samples = sampler.get_chain(flat=True)
        with contextlib.redirect_stdout(stream_to_logger("INFO")):  # type: ignore
            self.summary = self.emcee_summary(
                posterior_samples, true_values=true_values
            )
        self.raw_results = sampler

        # translate the results to a common data structure and return it
        var_names = self.problem.get_theta_names(tex=True, components=True)
        inference_data = az.from_emcee(sampler, var_names=var_names)
        return inference_data
