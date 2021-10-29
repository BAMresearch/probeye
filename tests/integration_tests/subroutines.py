# third party imports
import matplotlib.pyplot as plt

# local imports (inference engines)
from probeye.inference.scipy_.solver import ScipySolver
from probeye.inference.emcee_.solver import EmceeSolver
from probeye.inference.torch_.solver import PyroSolver

# local imports (post-processing)
from probeye.postprocessing.sampling import create_pair_plot
from probeye.postprocessing.sampling import create_posterior_plot
from probeye.postprocessing.sampling import create_trace_plot


def run_inference_engines(problem, true_values=None, n_steps=1000,
                          n_initial_steps=100, n_walkers=20, plot=True,
                          show_progress=True, run_scipy=True, run_emcee=True,
                          run_torch=True):
    """
    Runs a requested selection of inference engines on a given problem. This
    function is in a separate file to avoid repeating the same lines of code in
    all the integration tests.

    Parameters
    ----------
    problem : obj[InferenceProblem]
        Describes the inference problem including e.g. parameters and data.
    true_values : None, dict, optional
        Used for plotting 'true' parameter values. Keys are the parameter names
        and values are the values that are supposed to be shown in the marginal
        plots.
    n_steps : int, optional
        Number of steps (samples) to run.
    n_initial_steps : int, optional
        Number of steps for initial (burn-in) sampling.
    n_walkers : int, optional
        Number of walkers used by the estimator.
    plot : bool, optional
        If True, the data and the post-processing plots are plotted.
    show_progress : bool, optional
        If True, progress-bars will be shown, if available.
    run_scipy : bool, optional
        If True, the problem is solved with scipy (maximum likelihood estimate).
        Otherwise, no maximum likelihood estimate is derived.
    run_emcee : bool, optional
        If True, the problem is solved with the emcee solver. Otherwise, the
        emcee solver will not be used.
    run_torch : bool, optional
        If True, the problem is solved with the pyro/torch_ solver. Otherwise,
        the pyro/torch_ solver will not be used.
    """

    # this loop avoids to write down the same lines of code for the post-
    # processing for each inference engine again
    for inference_engine, requested_to_run in {'scipy': run_scipy,
                                               'emcee': run_emcee,
                                               'pyro':  run_torch}.items():

        # solve the problem with scipy (max. likelihood) if requested
        if inference_engine == 'scipy':
            is_sampling_solver = False
            if requested_to_run:
                scipy_solver = ScipySolver(problem, show_progress=show_progress)
                inference_data = scipy_solver.run_max_likelihood(
                    true_values=true_values)
            else:
                # in this case, the engine was not requested to run
                continue

        # solve the problem with emcee if requested
        elif inference_engine == 'emcee':
            is_sampling_solver = True
            if requested_to_run:
                emcee_solver = EmceeSolver(problem, show_progress=show_progress)
                inference_data = emcee_solver.run_mcmc(
                    n_walkers=n_walkers, n_steps=n_steps,
                    n_initial_steps=n_initial_steps)
            else:
                # in this case, the engine was not requested to run
                continue

        # solver the problem with pyro/torch if requested
        elif inference_engine == 'pyro':
            is_sampling_solver = True
            if requested_to_run:
                n_walkers_used = 1  # getting errors when trying to use more
                pyro_solver = PyroSolver(problem, show_progress=show_progress)
                inference_data = pyro_solver.run_mcmc(
                    n_walkers=n_walkers_used, n_steps=n_steps,
                    n_initial_steps=n_initial_steps)
            else:
                # in this case, the engine was not requested to run
                continue

        else:
            # if this happens, you messed something up in the defining
            # dictionary of this loop
            raise RuntimeError(
                f"Found unknown inference engine '{inference_engine}'!")

        # do the post-processing; note that the interface is the same for
        # each inference engine; also note, that the plots are not so much
        # intended for automatic testing, as for manually running the script
        if plot:
            if is_sampling_solver:
                create_pair_plot(inference_data, problem,
                                 true_values=true_values, show=False)
                create_posterior_plot(inference_data, problem,
                                      true_values=true_values, show=False)
                create_trace_plot(inference_data, problem, show=False)
                plt.show()  # shows all plots at once due to 'show=False' above
