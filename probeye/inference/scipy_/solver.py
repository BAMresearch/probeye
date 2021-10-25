# standard library imports
import copy as cp

# third party imports
import copy
import numpy as np
from scipy.optimize import minimize

# local imports
from probeye.inference.scipy_.priors import translate_prior
from probeye.inference.scipy_.noise_models import translate_noise_model


class ScipySolver:
    """Solver routines based on scipy and numpy for an InferenceProblem."""

    def __init__(self, problem, seed=1, verbose=True):
        """
        Parameters
        ----------
        problem : obj[InferenceProblem]
            Describes the inference problem including e.g. parameters and data.
        seed : int, optional
            Random state used for random number generation.
        verbose : bool, optional
            No logging output when False. More logging information when True.
        """

        # attributes from arguments
        self.verbose = verbose
        self.seed = seed

        # each noise model must be connected to the relevant experiment_names;
        # a copy is created before, so that this solver routine does not have
        # side effects on the original problem; such side effects would occur
        # due to calling the assign_experiments_to_noise_models-method below
        self.problem = cp.deepcopy(problem)
        self.problem.assign_experiments_to_noise_models()

        # translate the prior definitions to objects with computing capabilities
        self.priors = copy.deepcopy(self.problem.priors)
        for prior_name, prior_template in self.problem.priors.items():
            self.priors[prior_name] = translate_prior(prior_template)

        # translate the general noise model objects into solver specific ones
        self.noise_models = []
        for noise_model_base in self.problem.noise_models:
            self.noise_models.append(translate_noise_model(noise_model_base))

    def logprior(self, theta):
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
        for prior in self.priors.values():
            prms = self.problem.get_parameters(theta, prior.prms_def)
            lp += prior(prms, 'logpdf')
        return lp

    def sample_from_prior(self, prm_name, size):
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
        prior = self.priors[self.problem.parameters[prm_name].prior.name]
        # check for prior-priors; if a prior parameter is a latent
        # parameter and not a constant, one first samples from the prior
        # parameter's prior distribution, and then takes the mean of those
        # samples to sample from the first prior distribution; this procedure
        # is recursive, so that (in principle) one could also define priors of
        # the prior's prior parameters and so forth
        theta_aux = [0] * self.problem.parameters.n_latent_prms
        for prior_prm_name in prior.hyperparameters.keys():
            if self.problem.parameters[prior_prm_name].role == 'latent':
                samples = self.sample_from_prior(prior_prm_name, size)
                theta_aux[self.problem.parameters[prior_prm_name].index] =\
                    np.mean(samples)
        prms = self.problem.get_parameters(theta_aux, prior.hyperparameters)
        return prior.generate_samples(prms, size)

    def loglike(self, theta):
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
        for noise_model in self.noise_models:
            # compute the model response for the noise model's experiment_names
            model_response = self.problem.evaluate_model_response(
                theta, noise_model.experiment_names)
            # get the parameter values for the noise model's parameters
            prms_noise = self.problem.get_parameters(
                theta, noise_model.prms_def)
            # evaluate the loglike-contribution for the noise model
            ll += noise_model.loglike_contribution(model_response, prms_noise)
        return ll

    def run_max_likelihood(self, x0_dict=None, x0_prior='mean', default_x0=1.0,
                           true_values=None, method='Nelder-Mead',
                           solver_options=None, verbose=None):
        """
        Finds values for an InferenceProblem's latent parameters that maximize
        the problem's likelihood function. The used method is scipy's minimize
        function from the optimize submodule.

        Parameters
        ----------
        x0_dict : dict, optional
            Contains the start values for each latent variable. Via this arg
            the user can explicitly specify a start value for the optimization.
        x0_prior : {'mean', 'median'}, optional
            If x0_dict is not given, the start values will be derived from the
            priors, either using the 'mean' or 'median' value. If x0_dict is
            given, this argument has no effect.
        default_x0 : float, optional
            For uninformative priors, no mean or median value is defined. In
            those cases, the default_x0 value will be used as start value. If
            x0_dict is given, this argument has no effect.
        true_values : None, dict, optional
            Defines 'true' parameter values. Keys are the parameter names and
            values are the 'true' values. They are only used to print them next
            to the inferred parameter values from the optimization run.
        method : str, optional
            Defines the algorithm used by scipy.optimize.minimize. See the
            documentation of this scipy method to see all the options.
        solver_options : dict, optional
            Options passed to scipy.optimize.minimize under the 'options' key
            word argument. See the documentation of this scipy method to see
            available options.
        verbose : bool, optional
            No logging output when False. More logging information when True.

        Returns
        -------
        minimize_results : obj[scipy.optimize.optimize.OptimizeResult]
            An object returns by scipy's minimize function containing the
            optimization results. The parameter vector that optimizes the
            likelihood function can be requested via 'minimize_results.x'.
        """

        # allows to overwrite the default values the solver was initialized
        # with if this should be required
        if not verbose:
            verbose = self.verbose

        # since scipy's minimize function is used, we need a function that
        # returns the negative log-likelihood function (minimizing the negative
        # log-likelihood is equivalent to maximizing the (log-)likelihood)
        def fun(x):
            return -1 * self.loglike(x)

        # prepare the start value either from the given x0_dict or from the mean
        # values of the latent parameter's priors
        x0 = np.zeros(self.problem.n_latent_prms)
        if x0_dict:
            # in this case, the user explicitly defined the start values
            for prm_name, prm_value in x0_dict.items():
                idx = self.problem.parameters[prm_name].index
                x0[idx] = prm_value
        else:
            # in this case, the start values are derived from the priors; if
            # a prior i not uninformative, its mean value will be used; if a
            # prior is uninformative, the x0_default value will be used
            x0_dict = {}
            prms = cp.copy(self.problem.constant_prms_dict)
            for prm_name in self.problem.get_theta_names():
                prior_name = self.problem.parameters[prm_name].prior.name
                prior_type = self.problem.parameters[prm_name].prior.prior_type
                idx = self.problem.parameters[prm_name].index
                if prior_type != 'uninformative':
                    prm_value = self.priors[prior_name](
                        prms, x0_prior, use_ref_prm=False)
                    prms[prm_name] = prm_value
                    x0[idx] = prm_value
                else:
                    # no mean value can be requested if the prior is
                    # uninformative, hence a default value is used
                    x0[idx] = default_x0
                x0_dict[prm_name] = x0[idx]

        # this is the where the solver does its thing
        minimize_results = minimize(
            fun, x0, method=method, options=solver_options)

        # some convenient printout with respect to the solver's results
        if verbose:
            n_char_message = len(minimize_results.message)
            msg = (f"\nMaximum likelihood estimation (scipy)\n"
                   f"{'═' * n_char_message}\n"
                   f"{minimize_results.message}\n"
                   f"{'-' * n_char_message}\n"
                   f"Number of iterations:           {minimize_results.nit}\n"
                   f"Number of function evaluations: {minimize_results.nfev}\n"
                   f"{'-' * n_char_message}")
            print(msg)
            if minimize_results.success:
                theta_names = self.problem.get_theta_names(tex=False)
                n_char = max([len(name) for name in theta_names]) + 4
                for i, theta_name in enumerate(theta_names):
                    opt_name = f"{theta_name}_opt"
                    line = f"{opt_name:{n_char}s} = {minimize_results.x[i]:.6f}"
                    if true_values:
                        line += (f" (true = {true_values[theta_name]}, "
                                 f"start = {x0_dict[theta_name]})")
                    else:
                        line += f" (start = {x0_dict[theta_name]})"
                    print(line)
            print('')  # empty line at the end

        return minimize_results
