# ============================================================================ #
#                                   Imports                                    #
# ============================================================================ #

# third party imports
import matplotlib.pyplot as plt

# ============================================================================ #
#                           Taralli sampling solver                            #
# ============================================================================ #

def run_taralli_postprocessing(problem, emcee_model, plot=True, summary=True):
    """
    Solves an inference problem described in problem via taralli's Markov chain
    Monte Carlo method from EmceeParameterEstimator.

    Parameters
    ----------
    problem : obj[InferenceProblem]
        Describes the inference problem including e.g. parameters and data.
    emcee_model : obj[EmceeParameterEstimator]
        Contains the results of the sampling procedure.
    plot : bool, optional
        If True, the posterior's 1D and 2D marginal distributions are plotted in
        a triangular plot. If False, no plot will be created.
    summary : bool, optional
        If True, a summary of the posterior distribution is printed to the
        console. If False, no summary will be printed.
    """

    # plot the results
    if plot:
        tex_labels = problem.get_theta_names(tex=True)
        emcee_model.plot_posterior(dim_labels=tex_labels)
        plt.show(block=True)

    # prints a summary to the console
    if summary:
        emcee_model.summary()
