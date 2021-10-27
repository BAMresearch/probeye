# standard library imports
import time
import random

# third party imports
import numpy as np
import emcee
import arviz as az
from loguru import logger

# local imports
from probeye.subroutines import pretty_time_delta
from probeye.inference.scipy_.solver import ScipySolver


class EmceeSolver(ScipySolver):
    """
    Provides emcee-sampler which is a pure-Python implementation of Goodman &
    Weare’s Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler,
    see https://emcee.readthedocs.io/en/stable/.
    """
    def __init__(self, problem, seed=1, verbose=True):
        """See docstring of ScipySolver for information on the arguments."""
        logger.info("Initializing EmceeSolver")
        # initialize the scipy-based solver (ScipySolver)
        super().__init__(problem, seed=seed, verbose=verbose)

    def run_mcmc(self, n_walkers=20, n_steps=1000, n_initial_steps=100,
                 **kwargs):
        """
        Runs the emcee-sampler for the InferenceProblem the EmceeSolver was
        initialized with and returns the results as an arviz InferenceData obj.

        Parameters
        ----------
        n_walkers : int, optional
            Number of walkers used by the estimator.
        n_steps : int, optional
            Number of steps to run.
        n_initial_steps : int, optional
            Number of steps for initial (burn-in) sampling.
        **kwargs : optional
            Additional key-word arguments channeled to emcee.EnsembleSampler.

        Returns
        -------
        inference_data : obj[arviz.data.inference_data.InferenceData]
            Contains the results of the sampling procedure.
        """

        # draw initial samples from the parameter's priors
        logger.info("Drawing initial samples")
        sampling_initial_positions = np.zeros(
            (n_walkers, self.problem.n_latent_prms))
        theta_names = self.problem.get_theta_names()
        for i, parameter_name in enumerate(theta_names):
            sampling_initial_positions[:, i] = self.sample_from_prior(
                parameter_name, n_walkers)

        # The following code is based on taralli and merely adjusted to the
        # variables in the probeye setup; see https://gitlab.com/tno-bim/taralli

        # .................................................................... #
        #                             Pre-process                              #
        # .................................................................... #

        n_rows, n_cols = sampling_initial_positions.shape
        n_latent_prms = self.problem.n_latent_prms

        if n_cols != self.problem.n_latent_prms:
            raise ValueError(
                f"'sampling_initial_positions' should have {n_latent_prms} "
                f"columns (one for each latent parameter), but {n_cols} are "
                f"provided.")

        if n_rows != n_walkers:
            raise ValueError(
                f"'sampling_initial_positions' should have {n_walkers} rows "
                f"(one for each walker), but {n_rows} are provided.")

        random.seed(self.seed)
        np.random.seed(self.seed)
        rstate = np.random.mtrand.RandomState(self.seed)

        logger.info("Setting up EnsembleSampler")
        sampler = emcee.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=self.problem.n_latent_prms,
            log_prob_fn=lambda x: self.logprior(x) + self.loglike(x),
            **kwargs)

        sampler.random_state = rstate

        # .................................................................... #
        #    Initial sampling, burn-in: used to avoid a poor starting point    #
        # .................................................................... #

        logger.info("Starting sampling (initial + main)")
        start = time.time()
        state = sampler.run_mcmc(
            initial_state=sampling_initial_positions,
            nsteps=n_initial_steps,
            progress=self.verbose)
        sampler.reset()

        # .................................................................... #
        #                      Sampling of the posterior                       #
        # .................................................................... #
        sampler.run_mcmc(
            initial_state=state, nsteps=n_steps, progress=self.verbose)
        end = time.time()
        runtime_str = pretty_time_delta(end - start)
        logger.info(
            f"Sampling of the posterior distribution completed: {n_steps} steps"
            f" and {n_walkers} walkers.")
        logger.info(
            f"Total run-time (including initial sampling): {runtime_str}.")
        self.raw_results = sampler

        # translate the results to a common data structure and return it
        var_names = self.problem.get_theta_names(tex=True)
        inference_data = az.from_emcee(sampler, var_names=var_names)
        return inference_data
