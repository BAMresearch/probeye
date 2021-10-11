# ============================================================================ #
#                                   Imports                                    #
# ============================================================================ #

# standard library imports
import copy as cp

# third party imports
import numpy as np
from pyro.infer import NUTS, MCMC
import arviz as az

# local imports
from probeye.inference.torch_.priors import translate_prior_template
from probeye.inference.torch_.noise_models import translate_noise_model

# ============================================================================ #
#                          pyro/torch sampling solver                          #
# ============================================================================ #

def run_pyro_solver(problem_ori, n_walkers=1, n_steps=300, n_initial_steps=30,
                    step_size=0.1, verbose=True, use_gradients=True):
    """
    Solves an inference problem described in problem using pyro/torch.

    Parameters
    ----------
    problem_ori : obj[InferenceProblem]
        Describes the inference problem including e.g. parameters and data.
    n_walkers : int, optional
        Number of walkers used by the estimator.
    step_size : float, optional
        The step size of the NUTS sampler.
    n_steps: int, optional
        The number of steps the sampler takes.
    n_initial_steps: int, optional
        The number of steps for the burn-in phase.
    verbose : bool, optional
        No logging output when False. More logging information when True.
    use_gradients : bool, optional
        When True, torch's autograd functionality is used. In this case, the
        provided model has to accept tensor-valued input. If this value is set
        to False, the latent-parameter vector will be detached from torch's
        computational graph before evaluating the likelihood, which makes the
        sampling procedure extremely slow.
        
    Returns
    -------
    inference_data : obj[arviz.data.inference_data.InferenceData]
        Contains the results of the sampling procedure.
    """

    # if gradients should be used the problem's data structures have to be
    # converted into tensors; note that the convert-method does not modify the
    # original problem
    if use_gradients:
        problem = problem_ori.convert_data_to_tensor()
    else:
        # a copy is created here, so that this solver routine does not have side
        # effects on the original problem; such side effects would occur due to
        # calling the assign_experiments_to_noise_models-method below
        problem = cp.deepcopy(problem_ori)

    # each noise model must be connected to the relevant experiment_names
    problem.assign_experiments_to_noise_models()

    # the dictionary dependency_dict will contain all latent parameter names as
    # keys; the value of each key will be a list with latent hyperparameters of
    # the latent parameter's prior
    dependency_dict = {}
    for prm_name in problem.parameters.latent_prms:
        dependency_dict[prm_name] = []
        hyperparameters = problem.parameters[prm_name].prior.hyperparameters
        for prior_prm_name in hyperparameters:
            if prior_prm_name in problem.parameters.latent_prms:
                dependency_dict[prm_name].append(prior_prm_name)

    # this makes sure that the items in dependency are in an order that they can
    # be sampled from beginning (index 0) sequentially until the last item
    # without encountering any dependency problems (i.e. that another parameter
    # has to be sampled before the considered parameter can be sampled)
    consistent = False
    while not consistent:
        consistent = True
        idx_latent_dependencies =\
            [i for i, v in enumerate(dependency_dict.values()) if len(v) > 0]
        for idx in idx_latent_dependencies:
            key_idx = [*dependency_dict.keys()][idx]
            for dependency in dependency_dict[key_idx]:
                if key_idx in dependency_dict[dependency]:
                    raise RuntimeError(
                        f"Found circular dependency between {key_idx} and "
                        f"{dependency}!")
                idx_dependency = [*dependency_dict.keys()].index(dependency)
                if idx_dependency > idx:
                    consistent = False
                    tuples = list(dependency_dict.items())
                    tuples[idx], tuples[idx_dependency] = \
                        tuples[idx_dependency], tuples[idx]
                    dependency_dict = dict(tuples)

    # translate the prior definitions to objects with computing capabilities
    priors = {}
    for prm_name in dependency_dict.keys():
        prior_template = problem.parameters[prm_name].prior
        priors[prior_template.ref_prm] = \
            translate_prior_template(prior_template)

    # translate the general noise model objects into solver specific ones
    noise_models = []
    for noise_model_base in problem.noise_models:
        noise_models.append(translate_noise_model(
            noise_model_base, use_gradients=use_gradients))

    def get_theta_samples():
        """
        Provides a list of latent-parameter samples in form of torch.Tensors.

        Returns
        -------
        list[torch.Tensor]
            The sampled values based on the latent parameter's priors.
        """
        # even if a list is returned by this function, we initialize a dict here
        # to enable the action in the if-statement in the loop below
        pyro_parameter_samples = {}
        # we have to sample each parameter for which a prior is defined
        for ref_prm, prior_obj in priors.items():
            hyperprms_dict = {}
            for name in prior_obj.hyperparameters.keys():
                if name in pyro_parameter_samples:
                    hyperprms_dict[name] = pyro_parameter_samples[name]
                else:
                    # this is the default case, where the prior's parameters
                    # (i.e. the hyperparameters) are simply constants
                    hyperprms_dict[name] = problem.parameters[name].value
            # this is where the parameter's sample is generated with pyro
            pyro_parameter_samples[ref_prm] = prior_obj.sample(hyperprms_dict)
        return [*pyro_parameter_samples.values()]

    def loglike(theta_tensor_list):
        """
        Evaluates the log-likelihood function of the problem at theta.

        Parameters
        ----------
        theta_tensor_list : list[torch.Tensor]
            A vector of pyro.samples (i.e. tensors) for which the log-likelihood
            function should be evaluated.

        Returns
        -------
        ll : torch.Tensor
            The evaluated log-likelihood function for the given theta-vector.
        """
        # read values from the tensor-valued input; if gradients should not be
        # used, the theta-tensors need to be detached from the graph
        if use_gradients:
            theta = theta_tensor_list
        else:
            # this is where the detaching happens which will break performance
            theta = np.array([t.detach().numpy() for t in theta_tensor_list])
        # compute the contribution to the log-likelihood function for each noise
        # model and sum it all up
        for noise_model in noise_models:
            # compute the model response for the noise model's experiment_names
            model_response = problem.evaluate_model_response(
                theta, noise_model.experiment_names)
            # get the tensors for the noise model's parameters
            prms_noise = problem.get_parameters(theta, noise_model.prms_def)
            # evaluate the loglike-contribution for the noise model
            noise_model.sample_cond_likelihood(model_response, prms_noise,
                                               use_gradients)

    def posterior_model():
        """
        Returns the sampled log-likelihood in form of a torch.Tensor.

        Returns
        -------
        list[torch.Tensor]
            The sampled values based on the latent parameter's priors.
        """
        theta = get_theta_samples()
        return loglike(theta)

    # perform the sampling with the requested parameters
    kernel = NUTS(posterior_model, step_size=step_size, jit_compile=False)
    mcmc = MCMC(kernel, num_samples=n_steps, warmup_steps=n_initial_steps,
                num_chains=n_walkers, disable_progbar=not verbose)
    mcmc.run()

    # translate the results to a common data structure and return it
    var_names = problem.get_theta_names(tex=False)
    var_names_tex = problem.get_theta_names(tex=True)
    name_dict = {var_name: var_name_tex for var_name, var_name_tex
                 in zip(var_names, var_names_tex)}
    inference_data = az.from_pyro(mcmc)
    inference_data.rename(name_dict, groups='posterior', inplace=True)

    return inference_data
