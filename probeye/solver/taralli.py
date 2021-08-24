# ============================================================================ #
#                                   Imports                                    #
# ============================================================================ #

# third party imports
import numpy as np
import matplotlib.pyplot as plt

# local imports
from taralli.parameter_estimation.base import EmceeParameterEstimator

# ============================================================================ #
#                           Taralli sampling solver                            #
# ============================================================================ #

def taralli_solver(problem, n_walkers=20, n_steps=1000, plot=True,
                   summary=True):
    """
    Solves an inference problem described in problem via taralli's Markov chain
    Monte Carlo method from EmceeParameterEstimator.

    Parameters
    ----------
    problem : obj[InferenceProblem]
        Describes the inference problem including e.g. parameters and data.
    n_walkers : int, optional
        Number of walkers used by the estimator.
    n_steps : int, optional
        Number of steps to run.
    plot : bool, optional
        If True, the posterior's 1D and 2D marginal distributions are plotted in
        a triangular plot. If False, no plot will be created.
    summary : bool, optional
        If True, a summary of the posterior distribution is printed to the
        console. If False, no summary will be printed.

    Returns
    -------
    emcee_model : obj[EmceeParameterEstimator]
        Contains the results of the sampling procedure.
    """

    # draw initial samples from the parameter's priors
    init_array = np.zeros((n_walkers, problem.n_calibration_prms))
    theta_names = problem.get_theta_names()
    for i, prm_name in enumerate(theta_names):
        init_array[:, i] = problem.sample_from_prior(prm_name, n_walkers)

    # define the sampling task
    emcee_model = EmceeParameterEstimator(
        log_likelihood=problem.loglike,
        log_prior=problem.logprior,
        ndim=problem.n_calibration_prms,
        nwalkers=n_walkers,
        sampling_initial_positions=init_array,
        nsteps=n_steps
    )

    # perform the sampling
    emcee_model.estimate_parameters()

    # plot the results
    if plot:
        tex_labels = problem.get_theta_names(tex=True)
        emcee_model.plot_posterior(dim_labels=tex_labels)
        plt.show(block=True)

    # prints a summary to the console
    if summary:
        emcee_model.summary()

    return emcee_model
