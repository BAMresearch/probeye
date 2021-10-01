# local imports (inference engines)
from probeye.inference.emcee_.solver import run_emcee_solver
from probeye.inference.torch_.solver import run_pyro_solver

# local imports (post-processing)
from probeye.postprocessing.sampling import create_pair_plot
from probeye.postprocessing.sampling import create_posterior_plot
from probeye.postprocessing.sampling import create_trace_plot


def run_inference_engines(problem, n_steps=1000, n_initial_steps=100,
                          n_walkers=20, plot=True, verbose=True, run_emcee=True,
                          run_torch=True):
    """
    Runs a requested selection of inference engines on a given problem. This
    function is in a separate file to avoid repeating the same lines of code in
    all the integration tests.

    Parameters
    ----------
    problem : obj[InferenceProblem]
        Describes the inference problem including e.g. parameters and data.
    n_steps : int, optional
        Number of steps (samples) to run.
    n_initial_steps : int, optional
        Number of steps for initial (burn-in) sampling.
    n_walkers : int, optional
        Number of walkers used by the estimator.
    plot : bool, optional
        If True, the data and the post-processing plots are plotted.
    verbose : bool, optional
        If True, additional information will be printed to the console.
    run_emcee : bool, optional
        If True, the problem is solved with the emcee solver. Otherwise, the
        emcee solver will not be used.
    run_torch : bool, optional
        If True, the problem is solved with the pyro/torch_ solver. Otherwise,
        the pyro/torch_ solver will not be used.
    """

    # this loop avoids to write down the same lines of code for the post-
    # processing for each inference engine again
    for inference_engine, requested_to_run in {'emcee': run_emcee,
                                               'pyro':  run_torch}.items():

        # solve the problem with emcee if requested
        if inference_engine == 'emcee':
            if requested_to_run:
                inference_data = run_emcee_solver(
                    problem, n_walkers=n_walkers, n_steps=n_steps,
                    n_initial_steps=n_initial_steps, verbose=verbose)
            else:
                # in this case, the engine was not requested to run
                continue

        # solver the problem with pyro/torch if requested
        elif inference_engine == 'pyro':
            if requested_to_run:
                n_walkers_used = 1  # getting errors when trying to use more
                inference_data = run_pyro_solver(
                    problem, n_walkers=n_walkers_used, n_steps=n_steps,
                    n_initial_steps=n_initial_steps, verbose=verbose)
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
            create_pair_plot(inference_data, problem)
            create_posterior_plot(inference_data, problem)
            create_trace_plot(inference_data, problem)
