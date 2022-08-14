# standard library imports
from typing import Tuple, Optional, TYPE_CHECKING
import copy as cp

# third party imports
import numpy as np
import scipy as sp
from scipy.optimize import minimize
from loguru import logger

# local imports
from probeye.inference.solver import Solver
from probeye.inference.scipy.priors import translate_prior
from probeye.inference.scipy.likelihood_models import translate_likelihood_model
from probeye.subroutines import print_dict_in_rows, vectorize_numpy_dict
from probeye.subroutines import synchronize_objects

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.inverse_problem import InverseProblem
    from probeye.definition.forward_model import ForwardModelBase


class ScipySolver(Solver):
    """
    Solver based on scipy and numpy for an InverseProblem. The ScipySolver contains the
    methods for log-prior and log-likelihood evaluation. For information on the
    arguments see :class:`~probeye.inference.solver.Solver`.
    """

    def __init__(
        self,
        problem: "InverseProblem",
        seed: Optional[int] = None,
        show_progress: bool = True,
    ):
        logger.debug("Initializing ScipySolver")
        super().__init__(problem, seed=seed, show_progress=show_progress)

    def _translate_parameters(self):
        """
        Translate the inverse problem's parameters as needed for this solver.
        """
        # translate the prior definitions to objects with computing capabilities
        logger.debug("Translate the problem's parameters")
        for prior_template in self.problem.priors.values():
            prm_name = prior_template.ref_prm
            self.problem.parameters[prm_name] = self.problem.parameters[
                prm_name
            ].changed_copy(
                prior=translate_prior(
                    self.problem.parameters[prior_template.ref_prm].prior
                )
            )
        # translate non-scalar constants to numpy-arrays
        for prm_name in self.problem.parameters:
            if self.problem.parameters[prm_name].is_const:
                if self.problem.parameters[prm_name].dim > 1:
                    self.problem.parameters[prm_name] = self.problem.parameters[
                        prm_name
                    ].changed_copy(
                        value=np.array(self.problem.parameters[prm_name].value)
                    )

    def _translate_experiments(self):
        """
        Translate the inverse problem's experiments as needed for this solver.
        """
        # each tuple in the sensor_data must be converted to a numpy.ndarray
        logger.debug("Translate the problem's experiments")
        for exp_name in self.problem.experiments:
            for sensor_name in self.problem.experiments[exp_name].sensor_data:
                v = self.problem.experiments[exp_name].sensor_data[sensor_name]
                if isinstance(v, tuple):
                    a = np.array(v)
                    self.problem.experiments[exp_name].sensor_data[sensor_name] = a

    def _translate_forward_models(self):
        """
        Translate the inverse problem's forward models as needed for this solver.
        """
        logger.debug("Translate the problem's forward models")
        for fwd_name, fwd_obj in self.problem.forward_models.items():

            # create a full forward model from its hull where the sensors are not yet
            # connected to the experimental data
            forward_model_hull = self.problem.forward_models[fwd_name]
            forward_model = forward_model_hull.__class__.__bases__[0](
                fwd_name, *fwd_obj.args, **fwd_obj.kwargs
            )
            synchronize_objects(
                forward_model,
                forward_model_hull,
                exclude_startswith=("__", "_", "experiment_names"),
            )

            # add the experiments to the created forward model object by connecting
            # them with the respective sensors
            for exp_name in forward_model_hull.experiment_names:
                sensor_data = self.problem.experiments[exp_name].sensor_data
                forward_model.connect_experimental_data_to_sensors(
                    exp_name, sensor_data
                )

            # this is a default preparation for increased efficiency
            forward_model.prepare_experimental_inputs_and_outputs()

            # finally, add the forward model to the problem
            self.problem.forward_models[fwd_name] = forward_model

    def _translate_likelihood_models(self):
        """
        Translate the inverse problem's likelihood models as needed for this solver.
        """

        logger.debug("Translate the problem's likelihood models")
        for like_name in self.problem.likelihood_models:

            # the likelihood model's forward model is still referencing the old (i.e.,
            # not-translated) forward model and needs to be reset to the updated one
            fwd_name = self.problem.likelihood_models[like_name].forward_model.name
            fwd_model = self.problem.forward_models[fwd_name]
            self.problem.likelihood_models[like_name].forward_model = fwd_model
            self.problem.likelihood_models[like_name].determine_output_lengths()

            # translate the likelihood model
            self.problem.likelihood_models[like_name] = translate_likelihood_model(
                self.problem.likelihood_models[like_name]
            )

    def evaluate_model_response(
        self, theta: np.ndarray, forward_model: "ForwardModelBase", experiment_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the model response for each forward model for the given parameter
        vector theta and the given experiments.

        Parameters
        ----------
        theta
            A numeric vector for which the model responses should be evaluated. Which
            parameters these numbers refer to can be checked by calling
            self.theta_explanation() once the problem is set up.
        forward_model
            The forward model that should be evaluated.
        experiment_name
            The experiment, the forward model should be evaluated for.

        Returns
        -------
        model_response_vector
            Vector of the model responses (concatenated over output sensors).
        residuals_vector
            Vector of the model residuals (concatenated over output sensors).
        """

        # prepare the input dictionary for the forward model call
        prms_model = self.problem.get_parameters(theta, forward_model.prms_def)
        exp_inp = forward_model.input_from_experiments[experiment_name]
        inp = {**exp_inp, **prms_model}  # adds the two dictionaries

        # evaluate the forward model and translate the result to a single vector
        model_response_dict = forward_model(inp)
        model_response_vector = vectorize_numpy_dict(model_response_dict)

        # compute the residuals by comparing to the experimental response
        exp_response_dict = forward_model.output_from_experiments[experiment_name]
        exp_response_vector = vectorize_numpy_dict(exp_response_dict)
        residuals_vector = exp_response_vector - model_response_vector

        return model_response_vector, residuals_vector

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
        for prior in self.problem.priors.values():
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
        prior_name = self.problem.parameters[prm_name].prior.name
        prior = self.problem.priors[prior_name]
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

        # check whether the values of the latent parameters are within their domains;
        # they can end up outside their domains for example during sampling, when a
        # parameter vector is proposed, that contains a value that is not within the
        # specified bounds of a parameter
        if not self.problem.check_parameter_domains(theta):
            return -np.inf

        # compute the contribution to the log-likelihood function for each likelihood
        # model and sum it all up
        ll = 0.0
        for likelihood_model in self.problem.likelihood_models.values():
            # compute the model response/residuals for the likelihood model's experiment
            response, residuals = self.evaluate_model_response(
                theta, likelihood_model.forward_model, likelihood_model.experiment_name
            )
            # get the parameter values for the likelihood model's parameters
            prms_likelihood = self.problem.get_parameters(
                theta, likelihood_model.prms_def
            )
            # evaluate the loglike-contribution for the likelihood model
            ll += likelihood_model.loglike(response, residuals, prms_likelihood)
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
            InverseProblem.get_theta_names().
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
                    prm_value = self.problem.priors[prior_name](
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

    def summarize_point_estimate_results(
        self,
        minimize_results: sp.optimize.OptimizeResult,
        true_values: Optional[dict],
        x0_dict: dict,
        estimate_type: str = "maximum likelihood estimation",
    ):
        """
        Prints a summary of the results of the maximum likelihood estimation. For an
        explanation of the arguments, check out the docstring of the
        self.run_max_likelihood-method.
        """

        # the first part of the summary contains process information
        n_char_message = len(minimize_results.message)
        msg = (
            f"Results of {estimate_type}\n"
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

    def _run_ml_or_map(
        self,
        x0_dict: Optional[dict] = None,
        x0_prior: str = "mean",
        x0_default: float = 1.0,
        true_values: Optional[dict] = None,
        method: str = "Nelder-Mead",
        solver_options: Optional[dict] = None,
        use_priors: bool = True,
    ) -> sp.optimize.OptimizeResult:
        """
        Finds values for an InverseProblem's latent parameters that maximize the
        problem's likelihood or likelihood * prior function. The used method is scipy's
        minimize function from the optimize submodule.

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
        use_priors
            When True, the priors are included in the objective function (MAP).
            Otherwise, the priors are not included (ML).


        Returns
        -------
        minimize_results
            An object returns by scipy's minimize function containing the optimization
            results. The parameter vector that optimizes the likelihood function can be
            requested via 'minimize_results.x'.
        """

        # since scipy's minimize function is used, we need a function that returns the
        # negative log-likelihood function (minimizing the negative log-likelihood is
        # equivalent to maximizing the (log-)likelihood)
        if use_priors:
            estimate_type = "maximum a-posteriori estimation"

            def fun(x):
                return -(self.loglike(x) + self.logprior(x))

        else:
            estimate_type = "maximum likelihood estimation"

            def fun(x):
                return -self.loglike(x)

        # log at beginning so that errors can be associated
        logger.info(f"Solving problem via {estimate_type}")

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
        self.summarize_point_estimate_results(
            minimize_results, true_values, x0_dict, estimate_type
        )

        return minimize_results


class MaxLikelihoodSolver(ScipySolver):
    """
    Solver for a maximum likelihood estimation. This class is separate from ScipySolver
    so that its main function can be triggered by a 'run'-method. For information on the
    arguments see :class:`~probeye.inference.solver.Solver`.
    """

    def __init__(
        self,
        problem: "InverseProblem",
        seed: Optional[int] = None,
        show_progress: bool = True,
    ):
        logger.debug(f"Initializing {self.__class__.__name__}")
        super().__init__(problem, seed=seed, show_progress=show_progress)

    def run(
        self,
        x0_dict: Optional[dict] = None,
        x0_prior: str = "mean",
        x0_default: float = 1.0,
        true_values: Optional[dict] = None,
        method: str = "Nelder-Mead",
        solver_options: Optional[dict] = None,
    ) -> sp.optimize.OptimizeResult:
        """
        Triggers a maximum likelihood estimation. For more information on the arguments
        check out :func:`probeye.inference.scipy.solver._run_ml_or_map`.
        """
        return self._run_ml_or_map(
            x0_dict,
            x0_prior,
            x0_default,
            true_values,
            method,
            solver_options,
            use_priors=False,
        )


class MaxPosteriorSolver(ScipySolver):
    """
    Solver for maximum a-posteriori estimation. This class is separate from ScipySolver
    so that its main function can be triggered by a 'run'-method. For information on the
    arguments see :class:`~probeye.inference.solver.Solver`.
    """

    def __init__(
        self,
        problem: "InverseProblem",
        seed: Optional[int] = None,
        show_progress: bool = True,
    ):
        logger.debug(f"Initializing {self.__class__.__name__}")
        super().__init__(problem, seed=seed, show_progress=show_progress)

    def run(
        self,
        x0_dict: Optional[dict] = None,
        x0_prior: str = "mean",
        x0_default: float = 1.0,
        true_values: Optional[dict] = None,
        method: str = "Nelder-Mead",
        solver_options: Optional[dict] = None,
    ) -> sp.optimize.OptimizeResult:
        """
        Triggers a maximum a-posteriori estimation. For more information on the args
        check out :func:`probeye.inference.scipy.solver._run_ml_or_map`.
        """
        return self._run_ml_or_map(
            x0_dict,
            x0_prior,
            x0_default,
            true_values,
            method,
            solver_options,
            use_priors=True,
        )
