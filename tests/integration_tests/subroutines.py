# standard library
from typing import Optional, TYPE_CHECKING

# third party imports
import matplotlib.pyplot as plt

# local imports (inference engines)
from probeye.inference.scipy.solver import MaxLikelihoodSolver, MaxPosteriorSolver
from probeye.inference.emcee.solver import EmceeSolver
from probeye.inference.dynesty.solver import DynestySolver

# local imports (post-processing)
from probeye.postprocessing.sampling_plots import create_pair_plot
from probeye.postprocessing.sampling_plots import create_posterior_plot
from probeye.postprocessing.sampling_plots import create_trace_plot

# local imports (knowledge-graph related)
from probeye.ontology.knowledge_graph_export import export_results_to_knowledge_graph

# imports only needed for type hints
if TYPE_CHECKING:
    from probeye.definition.inverse_problem import InverseProblem


def run_inference_engines(
    problem: "InverseProblem",
    true_values: Optional[dict] = None,
    n_steps: int = 1000,
    n_initial_steps: int = 100,
    n_walkers: int = 20,
    plot: bool = True,
    show_progress: bool = True,
    write_to_graph: bool = False,
    knowledge_graph_file: str = "",
    data_dir: str = "",
    run_scipy: bool = True,
    run_emcee: bool = True,
    run_dynesty: bool = True,
):
    """
    Runs a requested selection of inference engines on a given problem. This function is
    in a separate file in order to avoid repeating the same lines of code in all the
    integration tests.

    Parameters
    ----------
    problem
        Describes the inverse problem including e.g. parameters and data.
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
    write_to_graph
        Triggers the export of the solver results to a given knowledge graph.
    knowledge_graph_file
        The owl-file of the knowledge graph that should be written to in case
        'write_to_graph' is set to True.
    data_dir
        The directory used by the knowledge graph export routine to write the data to
        (for example the samples of sampling routines).
    run_scipy
        If True, the problem is solved with scipy (maximum likelihood estimate).
        Otherwise, no maximum likelihood estimate is derived.
    run_emcee
        If True, the problem is solved with the emcee solver. Otherwise, the
        emcee solver will not be used.
    run_dynesty
        If True, the problem is solved with the dynesty solver. Otherwise, the
        dynesty solver will not be used.
    """

    def create_plots(inference_data, problem_cp, true_values_cp):
        # the '_cp'-suffix is just added to prevent inner-scope/outer-scope warnings
        create_pair_plot(
            inference_data,
            problem_cp,
            plot_priors=True,
            focus_on_posterior=True,
            true_values=true_values_cp,
            show=False,
            title="plot_priors=True, focus_on_posterior=True (default)",
        )
        create_pair_plot(
            inference_data,
            problem_cp,
            plot_priors=True,
            focus_on_posterior=False,
            kind="hexbin",
            true_values=true_values_cp,
            show=False,
            figsize=(8, 8),
            marginal_kwargs={"kind": "hist", "hist_kwargs": {"bins": 10}},
            title="plot_priors=True, focus_on_posterior=False, figsize=(8, 8) "
            + "(hex + hist)",
        )
        create_pair_plot(
            inference_data,
            problem_cp,
            plot_priors=False,
            focus_on_posterior=False,
            kind="scatter",
            true_values=true_values_cp,
            show=False,
            marginal_kwargs={"kind": "hist", "hist_kwargs": {"bins": 10}},
            title="plot_priors=False, focus_on_posterior=False (scatter + hist)",
        )
        create_pair_plot(
            inference_data,
            problem_cp,
            plot_priors=False,
            focus_on_posterior=True,
            true_values=true_values_cp,
            show=False,
            title="plot_priors=False, focus_on_posterior=True",
        )
        create_posterior_plot(
            inference_data,
            problem_cp,
            true_values=true_values_cp,
            show=False,
            title="This is a posterior-plot",
        )
        create_posterior_plot(
            inference_data,
            problem_cp,
            true_values=true_values_cp,
            figsize=(10, 5),
            show=False,
            title="This is a posterior-plot (figsize = (10, 5))",
        )
        create_trace_plot(
            inference_data, problem_cp, show=False, title="This is a trace-plot"
        )
        create_trace_plot(
            inference_data,
            problem_cp,
            show=False,
            figsize=(8, 8),
            title="This is a trace-plot (figsize = (8, 8))",
        )
        if plot:
            plt.show()  # shows all plots at once due to 'show=False' above
        else:
            plt.close("all")

    if run_scipy:
        ml_solver = MaxLikelihoodSolver(problem, show_progress=show_progress)
        inference_data_ml = ml_solver.run(true_values=true_values)
        if write_to_graph:
            export_results_to_knowledge_graph(
                problem,
                inference_data_ml,
                knowledge_graph_file,
                data_dir=data_dir,
            )
        map_possible = True
        for prior_name, prior_template in problem.priors.items():
            if prior_template.prior_type == "uninformative":
                map_possible = False
        if map_possible:
            map_solver = MaxPosteriorSolver(problem, show_progress=show_progress)
            _ = map_solver.run(true_values=true_values)

    if run_emcee:
        emcee_solver = EmceeSolver(problem, show_progress=show_progress)
        inference_data_emcee = emcee_solver.run(
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            true_values=true_values,
        )
        if write_to_graph:
            export_results_to_knowledge_graph(
                problem,
                inference_data_emcee,
                knowledge_graph_file,
                data_dir=data_dir,
            )
        create_plots(inference_data_emcee, emcee_solver.problem, true_values)

    if run_dynesty:
        dynesty_solver = DynestySolver(problem, show_progress=show_progress)
        inference_data_dynesty = dynesty_solver.run(
            "static", nlive=250, true_values=true_values
        )
        if write_to_graph:
            export_results_to_knowledge_graph(
                problem,
                inference_data_dynesty,
                knowledge_graph_file,
                data_dir=data_dir,
            )
        create_plots(inference_data_dynesty, dynesty_solver.problem, true_values)
