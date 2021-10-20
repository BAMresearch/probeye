# ============================================================================ #
#                                   Imports                                    #
# ============================================================================ #

# standard library imports
import copy as cp
import time
import logging
import random

# third party imports
import copy
import numpy as np
import emcee
import arviz as az

# local imports
from probeye.subroutines import pretty_time_delta
from probeye.inference.emcee_.priors import translate_prior
from probeye.inference.emcee_.noise_models import translate_noise_model

# ============================================================================ #
#                            emcee sampling solver                             #
# ============================================================================ #

def run_emcee_solver(problem_ori, n_walkers=20, n_steps=1000,
                     n_initial_steps=100, seed=1, verbose=True, **kwargs):
    """
    Solves an inference problem described in problem using emcee.

    Parameters
    ----------
    problem_ori : obj[InferenceProblem]
        Describes the inference problem including e.g. parameters and data.
    n_walkers : int, optional
        Number of walkers used by the estimator.
    n_steps : int, optional
        Number of steps to run.
    n_initial_steps : int, optional
        Number of steps for initial (burn-in) sampling.
    seed : int, optional
        Random state used for random number generation.
    verbose : bool, optional
        No logging output when False. More logging information when True.
    **kwargs : optional
        Additional key-word arguments channeled to emcee.EnsembleSampler.

    Returns
    -------
    sampler : obj[emcee.EnsembleSample]
        Contains the results of the sampling procedure.
    """

    # each noise model must be connected to the relevant experiment_names;
    # a copy is created before, so that this solver routine does not have side
    # effects on the original problem; such side effects would occur due to
    # calling the assign_experiments_to_noise_models-method below
    problem = cp.deepcopy(problem_ori)
    problem.assign_experiments_to_noise_models()

    # translate the prior definitions to objects with computing capabilities
    priors = copy.deepcopy(problem.priors)
    for prior_name, prior_template in problem.priors.items():
        priors[prior_name] = translate_prior(prior_template)

    # translate the general noise model objects into solver specific ones
    noise_models = []
    for noise_model_base in problem.noise_models:
        noise_models.append(translate_noise_model(noise_model_base))

    def logprior(theta):
        """
        Evaluates the log-prior function of the problem at theta.

        Parameters
        ----------
        theta : array_like
            A numeric vector for which the log-likelihood function should be
            evaluated. Which parameters these numbers refer to can be checked
            by calling self.theta_explanation() once the problem is set up.

        Returns
        -------
        lp : float
            The evaluated log-prior function for the given theta-vector.
        """
        lp = 0.0
        for prior in priors.values():
            prms = problem.get_parameters(theta, prior.prms_def)
            lp += prior(prms, 'logpdf')
        return lp

    def sample_from_prior(prm_name, size):
        """
        Generates random samples from a parameter's prior distribution and
        returns the generated samples.

        Parameters
        ----------
        prm_name : string
            The name of the parameter the prior is associated with.
        size : int
            The number of random samples to be drawn.

        Returns
        -------
        numpy.ndarray
            The generated samples.
        """
        prior = priors[problem.parameters[prm_name].prior.name]
        # check for prior-priors; if a prior parameter is a latent
        # parameter and not a constant, one first samples from the prior
        # parameter's prior distribution, and then takes the mean of those
        # samples to sample from the first prior distribution; this procedure
        # is recursive, so that (in principle) one could also define priors of
        # the prior's prior parameters and so forth
        theta_aux = [0] * problem.parameters.n_latent_prms
        for prior_prm_name in prior.hyperparameters.keys():
            if problem.parameters[prior_prm_name].role == 'latent':
                samples = sample_from_prior(prior_prm_name, size)
                theta_aux[problem.parameters[prior_prm_name].index] =\
                    np.mean(samples)
        prms = problem.get_parameters(theta_aux, prior.hyperparameters)
        return prior.generate_samples(prms, size)

    def loglike(theta):
        """
        Evaluates the log-likelihood function of the problem at theta.

        Parameters
        ----------
        theta : array_like
            A numeric vector for which the log-likelihood function should be
            evaluated. Which parameters these numbers refer to can be checked
            by calling self.theta_explanation() once the problem is set up.

        Returns
        -------
        ll : float
            The evaluated log-likelihood function for the given theta-vector.
        """
        # compute the contribution to the log-likelihood function for each noise
        # model and sum it all up
        ll = 0.0
        for noise_model in noise_models:
            # compute the model response for the noise model's experiment_names
            model_response = problem.evaluate_model_response(
                theta, noise_model.experiment_names)
            # get the parameter values for the noise model's parameters
            prms_noise = problem.get_parameters(theta, noise_model.prms_def)
            # evaluate the loglike-contribution for the noise model
            ll += noise_model.loglike_contribution(model_response, prms_noise)
        return ll

    # draw initial samples from the parameter's priors
    sampling_initial_positions = np.zeros((n_walkers, problem.n_latent_prms))
    theta_names = problem.get_theta_names()
    for i, parameter_name in enumerate(theta_names):
        sampling_initial_positions[:, i] = sample_from_prior(
            parameter_name, n_walkers)

    # The following code is based on taralli: https://gitlab.com/tno-bim/taralli
    # and merely adjusted to the variables in the probeye setup

    # ........................................................................ #
    #                               Pre-process                                #
    # ........................................................................ #

    if sampling_initial_positions.shape[1] != problem.n_latent_prms:
        raise ValueError(
            f"`sampling_initial_positions` should have {problem.n_latent_prms} "
            f"columns (one for each parameter of interest), but "
            f"{sampling_initial_positions.shape[1]} are provided.")

    if sampling_initial_positions.shape[0] != n_walkers:
        raise ValueError(
            f"`sampling_initial_positions` should have {n_walkers} rows (one "
            f"for each walker), but"
            f"{sampling_initial_positions.shape[0]} are provided.")

    random.seed(seed)
    np.random.seed(seed)
    rstate = np.random.mtrand.RandomState(seed)

    sampler = emcee.EnsembleSampler(
        nwalkers=n_walkers,
        ndim=problem.n_latent_prms,
        log_prob_fn=lambda x: logprior(x) + loglike(x),
        **kwargs)

    sampler.random_state = rstate

    # ........................................................................ #
    #      Initial sampling, burn-in: used to avoid a poor starting point      #
    # ........................................................................ #

    start = time.time()
    state = sampler.run_mcmc(
        initial_state=sampling_initial_positions,
        nsteps=n_initial_steps,
        progress=verbose)
    end = time.time()

    logging.info(
        f"Initial sampling completed: {n_initial_steps} steps and {n_walkers} "
        f"walkers.\n Total run-time: {pretty_time_delta(end - start)}.")

    sampler.reset()

    # ........................................................................ #
    #                        Sampling of the posterior                         #
    # ........................................................................ #
    start = time.time()
    sampler.run_mcmc(initial_state=state, nsteps=n_steps, progress=verbose)
    end = time.time()
    runtime_str = pretty_time_delta(end - start)
    logging.info(
        f"Sampling of the posterior distribution completed: {n_steps} steps "
        f"and {n_walkers} walkers.\n Total run-time: {runtime_str}.")

    # translate the results to a common data structure and return it
    var_names = problem.get_theta_names(tex=True)
    inference_data = az.from_emcee(sampler, var_names=var_names)
    return inference_data
