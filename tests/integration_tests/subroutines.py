# standard library
from typing import Optional, TYPE_CHECKING

# third party imports
import matplotlib.pyplot as plt

# local imports (inference engines)
from probeye.inference.scipy_.solver import ScipySolver
from probeye.inference.emcee_.solver import EmceeSolver
from probeye.inference.torch_.solver import PyroSolver
from probeye.inference.dynesty_.solver import DynestySolver

# local imports (post-processing)
from probeye.postprocessing.sampling import create_pair_plot
from probeye.postprocessing.sampling import create_posterior_plot
from probeye.postprocessing.sampling import create_trace_plot

# imports only needed for type hints
if TYPE_CHECKING:
    from probeye.definition.inference_problem import InferenceProblem


def run_inference_engines(
    problem: "InferenceProblem",
    true_values: Optional[dict] = None,
    n_steps: int = 1000,
    n_initial_steps: int = 100,
    n_walkers: int = 20,
    plot: bool = True,
    show_progress: bool = True,
    run_scipy: bool = True,
    run_emcee: bool = True,
    run_torch: bool = True,
    run_dynesty: bool = True,
):
    """
    Runs a requested selection of inference engines on a given problem. This function is
    in a separate file in order to avoid repeating the same lines of code in all the
    integration tests.

    Parameters
    ----------
    problem
        Describes the inference problem including e.g. parameters and data.
    true_values
        Used for plotting 'true' parameter values. Keys are the parameter names and
        values are the values that are supposed to be shown in the marginal plots.
    n_steps
        Number of steps (samples) to run.
    n_initial_steps
        Number of steps for initial (burn-in) sampling.
    n_walkers
        Number of walkers used by the estimator.
    plot
        If True, the data and the post-processing plots are plotted.
    show_progress
        If True, progress-bars will be shown, if available.
    run_scipy
        If True, the problem is solved with scipy (maximum likelihood estimate).
        Otherwise, no maximum likelihood estimate is derived.
    run_emcee
        If True, the problem is solved with the emcee solver. Otherwise, the
        emcee solver will not be used.
    run_torch
        If True, the problem is solved with the pyro/torch_ solver. Otherwise, the
        pyro/torch_ solver will not be used.
    run_dynesty
        If True, the problem is solved with the dynesty solver. Otherwise, the
        dynesty solver will not be used.
    """

    def create_plots(inference_data, problem, true_values):
        create_pair_plot(
            inference_data,
            problem,
            true_values=true_values,
            show=False,
            title="plot_priors=True, focus_on_posterior=False (default)",
        )
        create_pair_plot(
            inference_data,
            problem,
            focus_on_posterior=True,
            kind="hexbin",
            true_values=true_values,
            show=False,
            marginal_kwargs={"kind": "hist", "hist_kwargs": {"bins": 10}},
            title="plot_priors=True, focus_on_posterior=True (hex + hist)",
        )
        create_pair_plot(
            inference_data,
            problem,
            plot_priors=False,
            kind="scatter",
            true_values=true_values,
            show=False,
            marginal_kwargs={"kind": "hist", "hist_kwargs": {"bins": 10}},
            title="plot_priors=False (scatter + hist)",
        )
        create_pair_plot(
            inference_data,
            problem,
            plot_priors=False,
            true_values=true_values,
            show=False,
            title="plot_priors=False",
        )
        create_posterior_plot(
            inference_data,
            problem,
            true_values=true_values,
            show=False,
            title="This is a posterior-plot",
        )
        create_trace_plot(
            inference_data, problem, show=False, title="This is a trace-plot"
        )
        if plot:
            plt.show()  # shows all plots at once due to 'show=False' above

    if run_scipy:
        scipy_solver = ScipySolver(problem, show_progress=show_progress)
        inference_data = scipy_solver.run_max_likelihood(true_values=true_values)
        # no post processing for scipy solver as that is no sampler

    if run_emcee:
        emcee_solver = EmceeSolver(problem, show_progress=show_progress)
        inference_data = emcee_solver.run_mcmc(
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            true_values=true_values,
        )
        create_plots(inference_data, problem, true_values)

    if run_torch:
        n_walkers_used = 1  # getting errors when trying to use more
        pyro_solver = PyroSolver(problem, show_progress=show_progress)
        inference_data = pyro_solver.run_mcmc(
            n_walkers=n_walkers_used,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
        )
        create_plots(inference_data, problem, true_values)

    if run_dynesty:
        dynesty_solver = DynestySolver(problem, show_progress=show_progress)
        inference_data = dynesty_solver.run_dynesty(
            "static", nlive=250, true_values=true_values
        )
        create_plots(inference_data, problem, true_values)
