# standard library imports
from typing import TYPE_CHECKING
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
from probeye.inference.scipy_.solver import ScipySolver
from probeye.subroutines import stream_to_logger, print_dict_in_rows

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.inference_problem import InferenceProblem


class DynestySolver(ScipySolver):
    """
    A static and dynamic nested parameter estimator. It facilitates the use of the
    python package `dynesty`. The default is set to a static parameter estimator.

    _Note:_ For full details on the `dynesty` library see:
        https://dynesty.readthedocs.io/en/latest/index.html.
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

    def prior_transform(self, theta: np.ndarray) -> np.ndarray:
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
        ppf
            The vector of ppfs for each prior distribution at theta.
        """
        ppf = []
        for prior in self.priors.values():
            prms = self.problem.get_parameters(theta, prior.prms_def)
            try:
                ppf.append(prior(prms, "ppf"))
            except AttributeError as e:
                # Assume to be multivariate and let it raise
                # exceptions, if not.
                loc = prms[f"loc_{prior.ref_prm}"]
                scale = prms[f"scale_{prior.ref_prm}"]
                x = prms[prior.ref_prm]

                # Assume the MVN to be uncorrelated, only diagonal nonzero
                i, j = np.nonzero(scale)
                assert np.all(i == j)

                for i in range(len(loc)):
                    ppf.append(norm.ppf(q=x[i], loc=loc[i], scale=scale[i, i]))
        return ppf

    def get_summary(self, posterior_samples: np.ndarray) -> dict:
        """
        Computes and prints a summary of the posterior samples containing mean, median,
        standard deviation, 5th percentile and 95th percentile. Note, that this method
        was based on code from the taralli package: https://gitlab.com/tno-bim/taralli.

        Parameters
        ----------
        posterior_samples
            The generated samples in an array with as many columns as there are latent
            parameters, and n rows, where n = n_chains * n_steps.

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
        **kwargs,
    ) -> az.data.inference_data.InferenceData:
        """
        Runs the dynesty-sampler for the InferenceProblem the DynestySolver was
        initialized with and returns the results as an arviz InferenceData obj.

        Parameters
        ----------
        estimation_method
            "dynamic" or "static"
        nlive
            number of live points
        kwargs
            Additional key-word arguments channeled to emcee.EnsembleSampler.

        Returns
        -------
        inference_data or dynesty sampler
            Contains the results of the sampling procedure.
        """
        start = time.time()

        logger.info(f"Solving problem using dynesty sampler with {kwargs}")

        # The following code is based on taralli and merely adjusted to the variables
        # in the probeye setup; see https://gitlab.com/tno-bim/taralli

        # ............................................................................ #
        #                                 Pre-process                                  #
        # ............................................................................ #
        rstate = np.random.RandomState(self.seed)

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
        logger.info("")
        logger.info("Summary of sampling results")

        self.raw_results = sampler.results

        weights = np.exp(sampler.results.logwt - sampler.results.logz[-1])
        samples = dynesty.utils.resample_equal(sampler.results.samples, weights)

        with contextlib.redirect_stdout(stream_to_logger("INFO")):  # type: ignore
            self.summary = self.get_summary(samples)

        var_names = self.problem.get_theta_names(tex=True, components=True)

        posterior_dict = {name: samples[:, i] for i, name in enumerate(var_names)}

        inference_data = az.convert_to_inference_data(posterior_dict)
        # TODO maybe add information like "source = dynesty", timestamp, etc ...
        return inference_data
