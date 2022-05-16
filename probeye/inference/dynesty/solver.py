# standard library imports
from typing import TYPE_CHECKING, Optional
import time
import random
import contextlib

# third party imports
import numpy as np
import dynesty
import arviz as az
from loguru import logger
from tabulate import tabulate
from scipy.stats import norm

# local imports
from probeye.subroutines import pretty_time_delta
from probeye.subroutines import check_for_uninformative_priors
from probeye.inference.scipy.solver import ScipySolver
from probeye.subroutines import stream_to_logger
from probeye.subroutines import extract_true_values

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.inverse_problem import InverseProblem


class DynestySolver(ScipySolver):
    """
    A static and dynamic nested parameter estimator. It facilitates the use of the
    python package `dynesty`. The default is set to a static parameter estimator.

    _Note:_ For full details on the `dynesty` library see:
        https://dynesty.readthedocs.io/en/latest/index.html.

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
        self, problem: "InverseProblem", seed: int = 1, show_progress: bool = True
    ):
        logger.debug("Initializing DynestySolver")
        # check that the problem does not contain a uninformative prior
        check_for_uninformative_priors(problem)
        # initialize the scipy-based solver (ScipySolver)
        super().__init__(problem, seed=seed, show_progress=show_progress)

    def prior_transform(self, theta: np.ndarray) -> list:
        """
        Evaluates the ppf of the prior distributions at theta.

        Parameters
        ----------
        theta
            A numeric vector for which the ppf should be evaluated.
            Which parameters these numbers refer to can be checked by calling self.
            theta_explanation() once the problem is set up.

        Returns
        -------
        qs
            The vector of quantiles for each prior distribution at theta.
        """
        qs = []
        for prior in self.priors.values():
            prms = self.problem.get_parameters(theta, prior.prms_def)
            try:
                qs.append(prior(prms, "ppf"))
            except AttributeError as e:
                # This branch is active when there is no `ppf` method in
                # the prior distribution. For the case of a multivariate
                # normal distribution, we implement a workaround.
                mean = prms[f"mean_{prior.ref_prm}"]
                cov = prms[f"cov_{prior.ref_prm}"]
                x = prms[prior.ref_prm]

                # Assume the MVN to be uncorrelated, only diagonal nonzero
                i, j = np.nonzero(cov)
                assert np.all(i == j)

                mvn_qs = norm.ppf(q=x, loc=mean, scale=np.sqrt(np.diagonal(cov)))
                qs += list(mvn_qs)

        return qs

    def get_summary(
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

    def run_dynesty(
        self,
        estimation_method: str = "dynamic",
        nlive: int = 250,
        true_values: Optional[dict] = None,
        **kwargs,
    ) -> az.data.inference_data.InferenceData:
        """
        Runs the dynesty-sampler for the InverseProblem the DynestySolver was
        initialized with and returns the results as an arviz InferenceData obj.

        Parameters
        ----------
        estimation_method
            "dynamic" or "static"
        nlive
            number of live points
        true_values
            True parameter values, if known.
        kwargs
            Additional key-word arguments channeled to emcee.EnsembleSampler.

        Returns
        -------
        inference_data or dynesty sampler
            Contains the results of the sampling procedure.
        """
        start = time.time()

        logger.info(
            f"Solving problem using dynesty sampler with keyword arguments: {kwargs}"
        )

        # The following code is based on taralli and merely adjusted to the variables
        # in the probeye setup; see https://gitlab.com/tno-bim/taralli

        # ............................................................................ #
        #                                 Pre-process                                  #
        # ............................................................................ #
        rstate = np.random.default_rng(self.seed)

        if estimation_method == "dynamic":
            sampler = dynesty.DynamicNestedSampler(
                loglikelihood=self.loglike,
                prior_transform=self.prior_transform,
                ndim=self.problem.n_latent_prms_dim,
                rstate=rstate,
                nlive=nlive,
            )
            sampler.run_nested(print_progress=self.show_progress, **kwargs)
        elif estimation_method == "static":
            sampler = dynesty.NestedSampler(
                loglikelihood=self.loglike,
                prior_transform=self.prior_transform,
                ndim=self.problem.n_latent_prms_dim,
                rstate=rstate,
                nlive=nlive,
            )
            sampler.run_nested(print_progress=self.show_progress, **kwargs)
        else:
            raise RuntimeError(
                "Choose 'dynamic' or 'static' as the estimation_method parameter!"
            )
        end = time.time()
        runtime_str = pretty_time_delta(end - start)
        logger.info(f"Total run-time: {runtime_str}.")
        logger.info("Resample weighted samples to equal samples for post")
        logger.info("processing. Access the original dynesty results via  .raw_results")
        weights = np.exp(sampler.results.logwt - sampler.results.logz[-1])
        samples = dynesty.utils.resample_equal(sampler.results.samples, weights)
        logger.info("")  # empty line for visual buffer

        logger.info("Summary of sampling results (dynesty)")
        with contextlib.redirect_stdout(stream_to_logger("INFO")):  # type: ignore
            self.summary = self.get_summary(samples, true_values=true_values)
        logger.info("")  # empty line for visual buffer
        logger.info(f"Posterior log evidence: {sampler.results.logz[-1]}")
        logger.info("")  # empty line for visual buffer

        var_names = self.problem.get_theta_names(tex=True, components=True)

        posterior_dict = {name: samples[:, i] for i, name in enumerate(var_names)}

        inference_data = az.convert_to_inference_data(posterior_dict)
        # TODO maybe add information like "source = dynesty", timestamp, etc ...

        self.raw_results = sampler.results

        return inference_data
