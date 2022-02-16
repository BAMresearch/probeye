# standard library imports
from typing import Union, List, Tuple, Optional, TYPE_CHECKING
import copy as cp

# third party imports
import copy
import numpy as np
import scipy as sp
from scipy.optimize import minimize
from loguru import logger

# local imports
from probeye.inference.scipy_.priors import translate_prior
from probeye.inference.scipy_.likelihood_models import translate_likelihood_model
from probeye.subroutines import print_dict_in_rows, make_list

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.inference_problem import InferenceProblem


class ScipySolver:
    """
    Solver routines based on scipy and numpy for an InferenceProblem.

    Parameters
    ----------
    problem
        Describes the inference problem including e.g. parameters and data.
    seed
        Random state used for random number generation.
    show_progress
        When True, the progress of a solver routine will be shown (for example as a
        progress-bar) if such a feature is available. Otherwise, the progress will
        not shown.
    """

    def __init__(
        self, problem: "InferenceProblem", seed: int = 1, show_progress: bool = True
    ):

        # log at beginning so that errors can be associated
        logger.debug("Initializing ScipySolver")

        # attributes from arguments
        self.problem = problem
        self.show_progress = show_progress
        self.seed = seed

        # the following attributes will be set after the solver was run
        self.raw_results = None
        self.summary = {}  # type: dict

        # translate the prior definitions to objects with computing capabilities
        logger.debug("Translate problem's priors")
        self.priors = copy.deepcopy(self.problem.priors)
        for prior_name, prior_template in self.problem.priors.items():
            self.priors[prior_name] = translate_prior(prior_template)

        # translate the general likelihood model objects into solver specific ones
        logger.debug("Translate problem's likelihood models")
        self.likelihood_models = []
        for likelihood_model_definition in self.problem.likelihood_models.values():
            self.likelihood_models.append(
                translate_likelihood_model(likelihood_model_definition)
            )

    def evaluate_model_response(
        self, theta: np.ndarray, experiment_names: Union[str, List[str], None] = None
    ) -> dict:
        """
        Evaluates the model response for each forward model for the given parameter
        vector theta and the given experiments.

        Parameters
        ----------
        theta
            A numeric vector for which the model responses should be evaluated. Which
            parameters these numbers refer to can be checked by calling
            self.theta_explanation() once the problem is set up.
        experiment_names
            Contains the names of all or some of the experiments added to the inference
            problem. If this argument is None (which is a common use case) then all
            experiments defined in the problem (self.experiments) are used. The names
            provided here define the experiments that the fwd. model is evaluated for.

        Returns
        -------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.
        """

        # if experiments is not further specified all experiments added to the problem
        # will be accounted for when computing the model error
        if experiment_names is None:
            experiment_names = [*self.problem.experiments.keys()]
        else:
            # make sure that a given string is converted into a list
            experiment_names = make_list(experiment_names)

        # first, loop over all forward models, and then, over all experiments that are
        # associated with the corresponding model
        model_response_dict = {}
        for fwd_name, forward_model in self.problem.forward_models.items():
            # get the model parameters for the considered forward model
            prms_model = self.problem.get_parameters(theta, forward_model.prms_def)
            # get all experiments referring to the considered forward model
            relevant_experiment_names = self.problem.get_experiment_names(
                forward_model_names=fwd_name, experiment_names=experiment_names
            )
            # evaluate the forward model for each relevant experiment
            for exp_name in relevant_experiment_names:
                exp_dict = self.problem.experiments[exp_name]
                # prepare the model input values from the experimental data
                sensor_values = exp_dict["sensor_values"]
                exp_inp = {
                    input_sensor.name: sensor_values[input_sensor.name]
                    for input_sensor in forward_model.input_sensors
                }
                inp = {**exp_inp, **prms_model}  # adds the two dictionaries
                # finally, evaluate the forward model for this experiment
                model_response_dict[exp_name] = forward_model(inp)

        return model_response_dict

    def logprior(self, theta: np.ndarray) -> float:
        """
        Evaluates the log-prior function of the problem at theta.

        Parameters
        ----------
        theta
            A numeric vector for which the log-likelihood function should be evaluated.
            Which parameters these numbers refer to can be checked by calling self.
            theta_explanation() once the problem is set up.

        Returns
        -------
        lp
            The evaluated log-prior function for the given theta-vector.
        """
        lp = 0.0
        for prior in self.priors.values():
            prms = self.problem.get_parameters(theta, prior.prms_def)
            lp += prior(prms, "logpdf")
        return lp

    def sample_from_prior(self, prm_name: str, size: int) -> np.ndarray:
        """
        Generates random samples from a parameter's prior distribution and returns the
        generated samples.

        Parameters
        ----------
        prm_name
            The name of the parameter the prior is associated with.
        size
            The number of random samples to be drawn.

        Returns
        -------
            The generated samples.
        """
        prior = self.priors[self.problem.parameters[prm_name].prior.name]
        # check for prior-priors; if a prior parameter is a latent parameter and not a
        # constant, one first samples from the prior parameter's prior distribution, and
        # then takes the mean of those samples to sample from the first prior
        # distribution; this procedure is recursive, so that (in principle) one could
        # also define priors of the prior's prior parameters and so forth
        theta_aux = np.zeros(self.problem.parameters.n_latent_prms)
        for prior_prm_name in prior.hyperparameters.keys():
            if self.problem.parameters[prior_prm_name].role == "latent":
                samples = self.sample_from_prior(prior_prm_name, size)
                theta_aux[self.problem.parameters[prior_prm_name].index] = np.mean(
                    samples
                )
        prms = self.problem.get_parameters(theta_aux, prior.hyperparameters)
        return prior.generate_samples(prms, size)

    def loglike(self, theta: np.ndarray) -> float:
        """
        Evaluates the log-likelihood function of the problem at theta.

        Parameters
        ----------
        theta
            A numeric vector for which the log-likelihood function should be evaluated.
            Which parameters these numbers refer to can be checked by calling self.
            theta_explanation() once the problem is set up.

        Returns
        -------
        ll
            The evaluated log-likelihood function for the given theta-vector.
        """
        # compute the contribution to the log-likelihood function for each likelihood
        # model and sum it all up
        ll = 0.0
        for likelihood_model in self.likelihood_models:
            # compute the model response for the likelihood model's experiment_names
            model_response = self.evaluate_model_response(
                theta, likelihood_model.experiment_names
            )
            # get the parameter values for the likelihood model's parameters
            prms_likelihood = self.problem.get_parameters(
                theta, likelihood_model.prms_def
            )
            # evaluate the loglike-contribution for the likelihood model
            ll += likelihood_model.loglike(model_response, prms_likelihood)
        return ll

    def get_start_values(
        self,
        x0_dict: Optional[dict] = None,
        x0_prior: str = "mean",
        x0_default: float = 1.0,
    ) -> Tuple[np.ndarray, dict]:
        """
        Derives the start values for the maximum likelihood optimization run. For an
        explanation of the arguments, see self.run_max_likelihood.

        Returns
        -------
        x0
            A numeric vector with the derived start values in the order of
            InferenceProblem.get_theta_names().
        x0_dict
            Keys are the latent parameters, while the keys are their start values.
        """

        # this is going to be the start vector
        x0 = np.zeros(self.problem.n_latent_prms_dim)

        if x0_dict:
            # in this case, the user explicitly defined the start values
            for prm_name, prm_value in x0_dict.items():
                idx = self.problem.parameters[prm_name].index
                idx_end = self.problem.parameters[prm_name].index_end
                x0[idx:idx_end] = prm_value
        else:
            # in this case, the start values are derived from the priors; if a prior is
            # not uninformative, its mean value will be used; if a prior is
            # uninformative, the x0_default value will be used
            x0_dict = {}
            prms = cp.copy(self.problem.constant_prms_dict)
            for prm_name in self.problem.get_theta_names():
                prior_name = self.problem.parameters[prm_name].prior.name
                prior_type = self.problem.parameters[prm_name].prior.prior_type
                idx = self.problem.parameters[prm_name].index
                idx_end = self.problem.parameters[prm_name].index_end
                dim = self.problem.parameters[prm_name].dim
                if prior_type != "uninformative":
                    prm_value = self.priors[prior_name](
                        prms, x0_prior, use_ref_prm=False
                    )
                    prms[prm_name] = prm_value
                    x0[idx:idx_end] = prm_value
                else:
                    # no mean value can be requested if the prior is uninformative,
                    # hence a default value is used
                    x0[idx:idx_end] = [x0_default] * dim
                # scalar values should not be saved as one-element-lists
                if dim == 1:
                    x0_dict[prm_name] = x0[idx]
                else:
                    x0_dict[prm_name] = x0[idx:idx_end]

        return x0, x0_dict

    def summarize_ml_results(
        self,
        minimize_results: sp.optimize.optimize.OptimizeResult,
        true_values: Optional[dict],
        x0_dict: dict,
    ):
        """
        Prints a summary of the results of the maximum likelihood estimation. For an
        explanation of the arguments, check out the docstring of the
        self.run_max_likelihood-method.
        """

        # the first part of the summary contains process information
        n_char_message = len(minimize_results.message)
        msg = (
            f"Results of maximum likelihood estimation\n"
            f"{'=' * n_char_message}\n"
            f"{minimize_results.message}\n"
            f"{'-' * n_char_message}\n"
            f"Number of iterations:           {minimize_results.nit}\n"
            f"Number of function evaluations: {minimize_results.nfev}\n"
            f"{'-' * n_char_message}"
        )

        # log the results with a level depending on the status flag
        logger.info("")
        if minimize_results.status == 0:
            for line in msg.split("\n"):
                logger.info(line)
        else:  # in this case something went wrong
            for line in msg.split("\n"):
                logger.warning(line)

        # the second part shows the actual results and compares them with the true
        # values (if given) and the start values
        if minimize_results.success:
            theta_names = self.problem.get_theta_names(tex=False)
            n_char = max([len(name) for name in theta_names]) + 4
            for theta_name in theta_names:
                idx = self.problem.parameters[theta_name].index
                idx_end = self.problem.parameters[theta_name].index_end
                opt_name = f"{theta_name}_opt"
                opt_val = minimize_results.x[idx:idx_end]
                line = f"{opt_name:{n_char}s} = {opt_val}"
                if true_values:
                    line += (
                        f" (true = {true_values[theta_name]}, "
                        f"start = {x0_dict[theta_name]})"
                    )
                else:
                    line += f" (start = {x0_dict[theta_name]})"
                logger.info(line)
        logger.info("")  # empty line for visual buffer

    def run_max_likelihood(
        self,
        x0_dict: Optional[dict] = None,
        x0_prior: str = "mean",
        x0_default: float = 1.0,
        true_values: Optional[dict] = None,
        method: str = "Nelder-Mead",
        solver_options: Optional[dict] = None,
    ) -> sp.optimize.optimize.OptimizeResult:
        """
        Finds values for an InferenceProblem's latent parameters that maximize the
        problem's likelihood function. The used method is scipy's minimize function from
        the optimize submodule.

        Parameters
        ----------
        x0_dict
            Contains the start values for each latent variable. Via this arg the user
            can explicitly specify a start value for the optimization.
        x0_prior
            If x0_dict is not given, the start values will be derived from the priors,
            either using the 'mean' or 'median' value. If x0_dict is given, this
            argument has no effect. Valid values are 'mean' and 'median'.
        x0_default
            For uninformative priors, no mean or median value is defined. In those
            cases, the default_x0 value will be used as start value. If x0_dict is
            given, this argument has no effect.
        true_values
            Defines 'true' parameter values. Keys are the parameter names and values are
            the 'true' values. They are only used to print them next to the inferred
            parameter values from the optimization run.
        method
            Defines the algorithm used by scipy.optimize.minimize. See the documentation
            of this scipy method to see all the options.
        solver_options
            Options passed to scipy.optimize.minimize under the 'options' keyword arg.
            See the documentation of this scipy method to see available options.

        Returns
        -------
        minimize_results
            An object returns by scipy's minimize function containing the optimization
            results. The parameter vector that optimizes the likelihood function can be
            requested via 'minimize_results.x'.
        """

        # log at beginning so that errors can be associated
        logger.info("Solving problem via maximum likelihood estimation")

        # since scipy's minimize function is used, we need a function that returns the
        # negative log-likelihood function (minimizing the negative log-likelihood is
        # equivalent to maximizing the (log-)likelihood)
        def fun(x):
            return -self.loglike(x)

        # prepare the start value either from the given x0_dict or from the mean values
        # of the latent parameter's priors
        logger.debug("Deriving start values")
        x0, x0_dict = self.get_start_values(
            x0_dict=x0_dict, x0_prior=x0_prior, x0_default=x0_default
        )
        logger.info("Using start values:")
        print_dict_in_rows(x0_dict, printer=logger.info, val_fmt=None)

        # this is the where the solver does its thing
        logger.info(f"Starting optimizer (using {method})")
        if solver_options:
            logger.info("Specified solver options:")
            print_dict_in_rows(solver_options, printer=logger.info)
        else:
            logger.info("No solver options specified")
        minimize_results = minimize(fun, x0, method=method, options=solver_options)

        # note that in this case, the raw solver result is identical with the return-
        # value of this method; however, for other solver they differ; hence, this
        # attribute is set here only for consistency reasons
        self.raw_results = minimize_results
        self.summary = {
            "success": minimize_results.success,
            "theta_opt": minimize_results.x,
        }

        # some convenient printout with respect to the solver's results
        self.summarize_ml_results(minimize_results, true_values, x0_dict)

        return minimize_results
