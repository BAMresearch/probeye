# ============================================================================ #
#                                   Imports                                    #
# ============================================================================ #

# third party imports
import numpy as np
from tabulate import tabulate

# ============================================================================ #
#                           Taralli sampling solver                            #
# ============================================================================ #

def run_emcee_postprocessing(problem, emcee_sampler, verbose=True):
    """
    Runs a simple post-processing (mean, median, etc.) on the sampling results
    provided by an emcee.EnsembleSample object. Note that postprocessing
    routines should be made inference-engine-independent in the future.

    Parameters
    ----------
    problem : obj[InferenceProblem]
        Describes the inference problem including e.g. parameters and data.
    emcee_sampler : obj[emcee.EnsembleSample]
        Contains the results of the sampling procedure.
    verbose : bool, optional
        If True, a summary of the posterior distribution is printed to the
        console. If False, no summary will be printed.

    Returns
    -------
    summary : dict
        Contains 'mean', 'median' and 'covariance' keys with the corresponding
        results with respect to the samples as values.
    """

    # prepare a dictionary with basic evaluations
    posterior_sample = emcee_sampler.get_chain(flat=True)
    quantiles = np.quantile(
        posterior_sample, np.array([.25, .50, .75, .95]), axis=0)
    quantiles = np.atleast_1d(quantiles)
    mean = np.atleast_1d(np.mean(posterior_sample, axis=0))
    summary = dict.fromkeys(["mean", "median", "covariance"])
    summary["median"] = quantiles[1]
    summary["covariance"] = np.atleast_2d(np.cov(posterior_sample.T))
    summary["mean"] = mean

    if verbose:

        col_names = ["", "mean", "median", "sd", "25%", "75%", "95%"]
        tab = np.hstack(
            (
                [[name] for name in problem.get_theta_names()],
                summary["mean"].reshape(-1, 1),
                quantiles[1, :].reshape(-1, 1),
                np.sqrt(np.diag(summary["covariance"])).reshape(-1, 1),
                quantiles[0, :].reshape(-1, 1),
                quantiles[2, :].reshape(-1, 1),
                quantiles[3, :].reshape(-1, 1),
            )
        )

        print(tabulate(tab, headers=col_names, floatfmt=".2f"))

    return summary
