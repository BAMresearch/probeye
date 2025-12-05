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
from abc import ABC, abstractmethod

# local imports
from probeye.subroutines import pretty_time_delta
from probeye.subroutines import check_for_uninformative_priors
from probeye.inference.scipy.solver import ScipySolver
from probeye.subroutines import stream_to_logger
from probeye.subroutines import print_dict_in_rows
from probeye.subroutines import extract_true_values

from multiprocessing import Pool  # pickling problem

# from multiprocessing.pool import ThreadPool as Pool # no pickling needed but no time effect
import os

os.environ["OMP_NUM_THREADS"] = "1"
logprob = None


# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.inverse_problem import InverseProblem


class EmceeSolver(ScipySolver):
    """
    Provides emcee-sampler which is a pure-Python implementation of Goodman & Weare’s
    Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler. For more
    information, check out https://emcee.readthedocs.io/en/stable/.

    Parameters
    ----------
    problem
        Describes the inverse problem including e.g. parameters and data.
    seed
        Random state used for random number generation.
    show_progress
        When True, the progress of a solver routine will be shown (for example as a
        progress-bar) if such a feature is available. Otherwise, the progress will
        not shown.
    """

    def __init__(
        self,
        problem: "InverseProblem",
        seed: Optional[int] = None,
        show_progress: bool = True,
    ):
        logger.debug(f"Initializing {self.__class__.__name__}")
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

    def _build_logprob(self):

        global logprob

        def logprob(x):
            # Skip loglikelihood evaluation if logprior is equal
            # to negative infinity
            logprior = self.logprior(x)
            if logprior == -np.inf:
                return logprior

            # Otherwise return logprior + loglikelihood
            return logprior + self.loglike(x)

        return logprob

    def run(
        self,
        n_walkers: int = 20,
        n_steps: int = 1000,
        n_initial_steps: int = 100,
        true_values: Optional[dict] = None,
        parallel: bool = False,
        n_processes: int = 4,
        cluster_mode: str = "classic",
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
        parallel
            If True, the sampling is done in parallel using multiprocessing.
        n_processes
            Number of processes to use for parallel sampling.
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

        logger.debug("Setting up EnsembleSampler")
        self.runner = RunnerFactory.create(
            mode=cluster_mode,
            solver=self,
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            parallel=parallel,
            n_processes=n_processes,
            **kwargs,
        )

        self.runner.run(initial_state=sampling_initial_positions, true_values=true_values)

        # translate the results to a common data structure and return it
        self.var_names = self.problem.get_theta_names(tex=True, components=True)
        inference_data = az.from_emcee(self.runner.sampler, var_names=self.var_names)
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

        self.runner.sampler.run_mcmc(
            initial_state=state, nsteps=n_steps, progress=self.show_progress
        )
        self.var_names = self.problem.get_theta_names(tex=True, components=True)
        inference_data = az.from_emcee(self.runner.sampler, var_names=self.var_names)
        return inference_data


class RunnerFactory:
    """
    Factory for selecting the appropriate Runner based on user-specified mode.
    Ensures modularity and backward compatibility.

    Available modes:
        - "basic"          : no clustering, no pruning
        - "cluster-prune"  : perform clustering + pruning
    """

    RUNNER_MAP = {
        "classic": "ClassicRunnerBackend",
        "cluster-prune": "ClusterPruningRunner",
    }

    @staticmethod
    def create(
        mode: str,
        solver,
        n_walkers: int,
        n_steps: int,
        n_initial_steps: int,
        parallel: bool,
        n_processes: int = 4,
        **kwargs,
    ):
        """
        Create a Runner instance based on mode.

        Parameters
        ----------
        mode : str
            One of: "classic", "cluster-prune".
            Defaults to "classic" if None or empty.
        solver : EmceeSolver
            The solver object that owns the logprob, priors, likelihood, etc.
        n_walkers : int
        n_steps : int
        n_initial_steps : int
        parallel : bool
        kwargs : dict
            Additional options forwarded to the runner.

        Returns
        -------
        BaseRunner
            A concrete runner instance.
        """
        if mode is None or mode == "":
            mode = "classic"

        mode = mode.lower().strip()

        if mode not in RunnerFactory.RUNNER_MAP:
            raise ValueError(
                f"Unknown emcee run mode '{mode}'. Valid modes: "
                f"{list(RunnerFactory.RUNNER_MAP.keys())}"
            )

        runner_class_name = RunnerFactory.RUNNER_MAP[mode]

        # Retrieve the class object from globals() (or import if needed)
        if runner_class_name not in globals():
            raise RuntimeError(
                f"Runner class '{runner_class_name}' not found. "
                f"Ensure it is defined before calling RunnerFactory."
            )

        runner_cls = globals()[runner_class_name]

        return runner_cls(
            solver=solver,
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            parallel=parallel,
            n_processes=n_processes,
            **kwargs,
        )


class BaseRunnerBackend(ABC):
    def __init__(
        self,
        solver,
        n_walkers,
        n_steps,
        n_initial_steps,
        parallel=False,
        n_processes=4,
        **kwargs,
    ):

        self.solver = solver
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.n_initial_steps = n_initial_steps
        self.parallel = parallel
        self.n_processes = n_processes

        self.sampler_kwargs = kwargs  # passed into emcee.EnsembleSampler

    # ------------------------------------------------------------------
    # CENTRALIZED: Parallel or serial sampler creation
    # ------------------------------------------------------------------
    def _create_sampler(self, logprob):
        """
        Creates an emcee EnsembleSampler, with or without multiprocessing.
        All runners share this.
        """

        if self.parallel:
            self.pool = Pool(processes=self.n_processes)
            return emcee.EnsembleSampler(
                nwalkers=self.n_walkers,
                ndim=self.solver.problem.n_latent_prms_dim,
                log_prob_fn=logprob,
                pool=self.pool,
                **self.sampler_kwargs,
            )
        else:
            self.pool = None
            return emcee.EnsembleSampler(
                nwalkers=self.n_walkers,
                ndim=self.solver.problem.n_latent_prms_dim,
                log_prob_fn=logprob,
                **self.sampler_kwargs,
            )

    # ------------------------------------------------------------------
    def close_pool(self):
        """Close pool if used."""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    @abstractmethod
    def run(self, initial_state=None):
        """Run the sampling procedure."""
        pass


class ClassicRunnerBackend(BaseRunnerBackend):
    """
    Classic runner backend that runs initial burn-in and main sampling in one go.
    """

    def run(self, initial_state=None, true_values=None):
        logprob = self.solver._build_logprob()

        self.sampler = self._create_sampler(logprob)

        state_burn_in = self.run_burn_in(initial_state)
        state_postproc_burn_in = self.postproc_burn_in(
            state_burn_in, true_values=true_values
        )
        state_main = self.run_main_sampling(state_postproc_burn_in)
        state_postproc_main = self.postprocess_main_sampling(
            state_main, true_values=true_values
        )

        self.close_pool()  # clean up if parallel
        return self.sampler, state_postproc_main

    def run_burn_in(self, initial_state=None):
        """Run burn-in phase."""
        state = self.sampler.run_mcmc(
            initial_state, self.n_initial_steps, progress=self.solver.show_progress
        )
        return state

    def postproc_burn_in(self, state, true_values=None):
        """Postprocess burn-in phase (no-op in classic)."""
        logger.info("")
        logger.info("Summary of sampling results during burn-in (emcee)")
        posterior_samples = self.sampler.get_chain(flat=True)
        with contextlib.redirect_stdout(stream_to_logger("INFO")):  # type: ignore
            self.solver.summary = self.solver.emcee_summary(
                posterior_samples, true_values=true_values
            )
        logger.info("")
        self.sampler.reset()
        return state

    def run_main_sampling(self, state):
        """Run main sampling phase."""
        start = time.time()
        self.sampler.run_mcmc(state, self.n_steps, progress=self.solver.show_progress)
        end = time.time()
        runtime_str = pretty_time_delta(end - start)
        logger.info(
            f"Sampling of the posterior distribution completed: {self.n_steps} steps and "
            f"{self.n_walkers} walkers."
        )
        logger.info(f"Total run-time (including initial sampling): {runtime_str}.")
        logger.info("")
        self.raw_results = self.sampler
        return state

    def postprocess_main_sampling(self, state, true_values=None):
        """Postprocess main sampling phase (no-op in classic)."""
        logger.info("Summary of sampling results during main sampling (emcee)")
        posterior_samples = self.sampler.get_chain(flat=True)
        with contextlib.redirect_stdout(stream_to_logger("INFO")):  # type: ignore
            self.solver.summary = self.solver.emcee_summary(
                posterior_samples, true_values=true_values
            )
        logger.info("")
        return state


class WalkerClusteringMixin:
    """
    Optional clustering/postprocessing for walker pruning.

    Drop-in mixin module that can be attached to EmceeSolver
    without modifying original logic unless called explicitly.
    """

    def prune_walkers_by_logp_jump(self, sampler, jump_factor=5.0, return_mask=False):
        """
        Detect a likelihood jump in sorted log-probabilities and remove walkers
        below the jump (cluster pruning).

        Parameters
        ----------
        sampler : emcee.EnsembleSampler
            The sampler object.
        jump_factor : float
            Jump threshold multiplier relative to median diff.
        return_mask : bool
            Whether to return a boolean mask for retained walkers.

        Returns
        -------
        kept : list[int]
        pruned : list[int]
        stats : dict
        mask : optional boolean array
        """
        final_logp = sampler.lnprobability[:, -1]
        nwalkers = len(final_logp)

        sorted_idx = np.argsort(final_logp)
        sorted_logp = np.nan_to_num(final_logp[sorted_idx])
        diffs = np.diff(sorted_logp)

        median_diff = np.median(diffs)
        jump_threshold = jump_factor * median_diff

        jumps = np.where(diffs > jump_threshold)[0]

        if len(jumps) == 0:
            kept = list(range(nwalkers))
            pruned = []
            stats = dict(
                logp_sorted=sorted_logp,
                diffs=diffs,
                jump_index=None,
                threshold_logp=None,
                n_kept=nwalkers,
                n_pruned=0,
            )
            return (
                (kept, pruned, stats, np.ones(nwalkers, dtype=bool))
                if return_mask
                else (kept, pruned, stats)
            )

        # take the largest jump
        j = jumps[-1]
        threshold = sorted_logp[j]

        mask = final_logp > threshold
        kept = np.where(mask)[0].tolist()
        pruned = np.where(~mask)[0].tolist()

        stats = dict(
            logp_sorted=sorted_logp,
            diffs=diffs,
            jump_index=j,
            threshold_logp=threshold,
            n_kept=len(kept),
            n_pruned=len(pruned),
        )

        if return_mask:
            return kept, pruned, stats, mask

        return kept, pruned, stats

    def log_cluster_stats(self, stats, kept, pruned):
        """Unified logging block."""
        logger.info("")
        logger.info("=== Walker Clustering Diagnostics ===")

        logger.info(f"Kept   : {stats['n_kept']}")
        logger.info(f"Pruned : {stats['n_pruned']}")

        if stats["threshold_logp"] is not None:
            logger.info(f"Threshold logp at jump: {stats['threshold_logp']:.3f}")
            j = stats["jump_index"]
            logger.info(f"Jump detected between sorted idx {j} and {j+1}")

        logger.info("====================================")

        kept_posterior_samples = self.sampler.get_chain(flat=False)[:, kept, :].reshape(
            -1, self.solver.problem.n_latent_prms
        )
        logger.info("")
        logger.info("Summary of sampling results after pruning (emcee)")
        with contextlib.redirect_stdout(stream_to_logger("INFO")):  # type: ignore
            summary = self.solver.emcee_summary(
                kept_posterior_samples, true_values=None
            )


class ClusterPruningRunnerBackend(ClassicRunnerBackend, WalkerClusteringMixin):
    """
    Runner backend that performs clustering and pruning after burn-in phase.
    """

    def postproc_burn_in(self, state):
        """Perform clustering and pruning after burn-in."""
        logger.info("")
        logger.info("Summary of sampling results during burn-in (emcee)")
        posterior_samples = self.sampler.get_chain(flat=True)
        with contextlib.redirect_stdout(stream_to_logger("INFO")):  # type: ignore
            self.solver.summary = self.solver.emcee_summary(
                posterior_samples, true_values=None
            )
        logger.info("")  # empty line for visual buffer
        kept, pruned, stats, mask = self.prune_walkers_by_logp_jump(
            sampler=self.sampler, jump_factor=5.0, return_mask=True
        )
        self.solver.log_cluster_stats(stats, kept, pruned)

        if stats["n_kept"] < 2:
            raise RuntimeError(
                "Too many walkers pruned; cannot continue MCMC. "
                "Reduce `clustering_jump_factor` or turn off clustering."
            )

        # rebuild initial state for the second run
        full_state = np.zeros_like(state.coords)
        full_state[mask] = state[mask]

        # resample pruned walkers
        rng = np.random.default_rng()
        kept_states = full_state[mask]
        for w in np.where(~mask)[0]:
            a, b = rng.choice(len(kept_states), 2, replace=False)
            wgt = rng.random()
            full_state[w] = wgt * kept_states[a] + (1 - wgt) * kept_states[b]

        return full_state

    def postprocess_main_sampling(self, state):
        logger.info("")
        logger.info("Summary of sampling results during burn-in (emcee)")
        posterior_samples = self.sampler.get_chain(flat=True)
        with contextlib.redirect_stdout(stream_to_logger("INFO")):  # type: ignore
            self.solver.summary = self.solver.emcee_summary(
                posterior_samples, true_values=None
            )
        logger.info("")  # empty line for visual buffer
        kept, pruned, stats, mask = self.prune_walkers_by_logp_jump(
            sampler=self.sampler, jump_factor=5.0, return_mask=True
        )
        self.solver.log_cluster_stats(stats, kept, pruned)

        return state[mask]
