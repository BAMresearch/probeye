# ============================================================================ #
#                                   Imports                                    #
# ============================================================================ #

# third party imports
import copy
import numpy as np

# local imports
from probeye.inference.taralli_.priors import translate_prior_template
from taralli.parameter_estimation.base import EmceeParameterEstimator

# ============================================================================ #
#                           Taralli sampling solver                            #
# ============================================================================ #

def run_taralli_solver(problem, n_walkers=20, n_steps=1000, **kwargs):
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
    **kwargs : optional
        Additional key-word arguments channeled to EmceeParameterEstimator.

    Returns
    -------
    emcee_model : obj[EmceeParameterEstimator]
        Contains the results of the sampling procedure.
    """

    # translate the prior definitions to objects with computing capabilities
    priors = copy.deepcopy(problem.priors)
    for prior_name, prior_template in problem.priors.items():
        priors[prior_name] = translate_prior_template(prior_template)

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
        # check for prior-priors; if a prior parameter is a calibration
        # parameter and not a constant, one first samples from the prior
        # parameter's prior distribution, and then takes the mean of those
        # samples to sample from the first prior distribution; this procedure
        # is recursive, so that (in principle) one could also define priors of
        # the prior's prior parameters and so forth
        theta_aux = [0] * problem.parameters.n_calibration_prms
        for prior_prm_name in prior.prms_def_no_ref.keys():
            if problem.parameters[prior_prm_name].role == 'calibration':
                samples = sample_from_prior(prior_prm_name, size)
                theta_aux[problem.parameters[prior_prm_name].index] =\
                    np.mean(samples)
        prms = problem.get_parameters(theta_aux, prior.prms_def_no_ref)
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

        # evaluate the model error for each defined forward model and each
        # output sensor in this/those forward model(s)
        model_error_dict = problem.evaluate_model_error(theta)

        # compute the contribution to the log-likelihood function for the
        # model error of forward model and output sensor, and sum it all up
        ll = 0.0
        for me_dict in model_error_dict.values():
            for sensor, me_vector in me_dict.items():
                noise_model = problem.noise_models[sensor]
                prms_noise = problem.get_parameters(theta, noise_model.prms_def)
                ll += noise_model.loglike_contribution(me_vector, prms_noise)

        return ll

    # draw initial samples from the parameter's priors
    init_array = np.zeros((n_walkers, problem.n_calibration_prms))
    theta_names = problem.get_theta_names()
    for i, parameter_name in enumerate(theta_names):
        init_array[:, i] = sample_from_prior(parameter_name, n_walkers)

    # define the sampling task
    emcee_model = EmceeParameterEstimator(
        log_likelihood=loglike,
        log_prior=logprior,
        ndim=problem.n_calibration_prms,
        nwalkers=n_walkers,
        sampling_initial_positions=init_array,
        nsteps=n_steps,
        **kwargs
    )

    # perform the sampling
    emcee_model.estimate_parameters()

    return emcee_model
