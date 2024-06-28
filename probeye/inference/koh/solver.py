# standard library imports
from typing import TYPE_CHECKING, Optional, Union, Callable, Tuple

# third party imports
import numpy as np
from loguru import logger
import emcee
import arviz as az
import time
import random
import contextlib
import chaospy #FIXME: This should not be always imported

# local imports
from probeye.subroutines import check_for_uninformative_priors
from probeye.inference.emcee.solver import EmceeSolver
from probeye.inference.koh.likelihood_models import translate_likelihood_model
from probeye.subroutines import vectorize_nd_numpy_dict, vectorize_numpy_dict, vectorize_tuple_pce_dict
from probeye.subroutines import pretty_time_delta
from probeye.subroutines import check_for_uninformative_priors
from probeye.subroutines import stream_to_logger
from probeye.subroutines import print_dict_in_rows


# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.inverse_problem import InverseProblem


class KOHSolver(EmceeSolver):
    """
    Provides a modification of emcee solver to work with Kennedy and O'Hagan's
    type of implementations.

    Parameters
    ----------
    problem
        Describes the inverse problem including e.g. parameters and data.
    seed
        Random state used for random number generation.
    show_progress
        When True, the progress of a solver routine will be shown (for example as a
        progress-bar) if such a feature is available. Otherwise, the progress will
        not shown.
    """

    def __init__(
        self,
        problem: "InverseProblem",
        seed: Optional[int] = None,
        show_progress: bool = True,
        extended_problem: bool = False,
        extension_variables: Optional[str] = None,
        scale_coordinates_flag: bool = False,
        scale_residuals: float = 1.0,
    ):
        logger.debug(f"Initializing {self.__class__.__name__}")
        # check that the problem does not contain a uninformative prior
        check_for_uninformative_priors(problem)
        # initialize the scipy-based solver (ScipySolver)
        super().__init__(problem, seed=seed, show_progress=show_progress)
        self.extended_problem = extended_problem
        self.scale_residuals = scale_residuals
        if self.extended_problem:
            if extension_variables is None:
                raise Exception(
                    "Extension variable must be specified if extended problem is used."
                )
            else:
                self.extension_variables = extension_variables

            self.scale_coordinates = (
                self.scale_coordinates if scale_coordinates_flag else lambda x: x
            )

            # The extended model needs the bias defined in the inverse problem
            if not hasattr(problem, "bias_model_class") and not hasattr(
                problem, "bias_parameters"
            ):
                raise Exception("The inverse problem must have a bias model defined.")

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

        # Formualtion for extended problem
        if self.extended_problem:
            residuals_list = []
            extension_coordinates = np.array([]).reshape(
                len(self.extension_variables), 0
            )
            ll = 0.0
            for likelihood_model in self.problem.likelihood_models.values():
                # compute the model response/residuals for the likelihood model's experiment
                response, residuals = self.evaluate_model_response(
                    theta,
                    likelihood_model.forward_model,
                    likelihood_model.experiment_name,
                )
                prms_likelihood = self.problem.get_parameters(
                    theta, likelihood_model.prms_def
                )
                residuals_list.append(residuals)
                # TODO: Doing this in the extension evaluation is very inefficient
                temp_extension_coordinates = np.array([])
                for variable in self.extension_variables:
                    try:
                        temp_extension_coordinates = np.vstack(
                            (
                                temp_extension_coordinates,
                                self.problem.experiments[
                                    likelihood_model.experiment_name
                                ].sensor_data[variable],
                            )
                        )
                    except ValueError:
                        temp_extension_coordinates = self.problem.experiments[
                            likelihood_model.experiment_name
                        ].sensor_data[variable]
                if len(temp_extension_coordinates.shape) == 1:
                    temp_extension_coordinates = temp_extension_coordinates.reshape(
                        1, -1
                    )
                extension_coordinates = np.hstack(
                    (extension_coordinates, temp_extension_coordinates)
                )
            # TODO: In future, bias should have its own model that allows for input/output definition
            #       For now, we assume that the bias is a GP that takes the extension variable as input
            bias = self.problem.bias_model_class(**self.problem.bias_parameters)
            bias.train(
                self.scale_coordinates(np.array(extension_coordinates).transpose()),
                np.concatenate(residuals_list) * self.scale_residuals,
            )

            # Save bias
            self.problem.bias_model = bias.clone_with_theta()
            # Return log_marginal_likelihood of the bias GP
            # TODO: This should be more flexible in the future

            return float(bias.gp.log_marginal_likelihood())

        # Formulation for standard problem
        else:
            # compute the contribution to the log-likelihood function for each likelihood
            # model and sum it all up
            ll = 0.0
            for likelihood_model in self.problem.likelihood_models.values():
                # compute the model response/residuals for the likelihood model's experiment
                response, residuals = self.evaluate_model_response(
                    theta,
                    likelihood_model.forward_model,
                    likelihood_model.experiment_name,
                )
                # get the parameter values for the likelihood model's parameters
                prms_likelihood = self.problem.get_parameters(
                    theta, likelihood_model.prms_def
                )
                # evaluate the loglike-contribution for the likelihood model
                ll += likelihood_model.loglike(response, residuals, prms_likelihood)
            return ll

    @staticmethod
    def scale_coordinates(arr):
        """
        Scales an n-dimensional numpy array of coordinates to the range [0, 1] based on the original min and max values present in the array.
        """

        # Iterate over dimensions and scale each dimension
        for dim in range(arr.shape[1]):
            min_val, max_val = arr[:, dim].min(), arr[:, dim].max()
            arr[:, dim] = (arr[:, dim] - min_val) / (max_val - min_val)

        return arr


class OGPSolver(KOHSolver):
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

        assert (
            self.problem.bias_parameters["computational_gp"] == "forward_model"
        ) or isinstance(
            self.problem.bias_parameters["computational_gp"], Callable
        ), "Computational GP must be either a callable or 'forward_model'"
        assert (
            self.problem.bias_parameters["derivative"] == "forward_model"
        ) or isinstance(
            self.problem.bias_parameters["derivative"], Callable
        ), "Derivative must be either a callable or 'forward_model'"
        # check whether the values of the latent parameters are within their domains;
        # they can end up outside their domains for example during sampling, when a
        # parameter vector is proposed, that contains a value that is not within the
        # specified bounds of a parameter

        if not self.problem.check_parameter_domains(theta):
            return -np.inf

        # Formulation for extended problem
        if self.extended_problem:
            residuals_list = []
            extension_coordinates = np.array([]).reshape(
                len(self.extension_variables), 0
            )
            ll = 0.0
            for likelihood_model in self.problem.likelihood_models.values():
                # compute the model response/residuals for the likelihood model's experiment
                response, residuals = self.evaluate_model_response(
                    theta,
                    likelihood_model.forward_model,
                    likelihood_model.experiment_name,
                )
                prms_likelihood = self.problem.get_parameters(
                    theta, likelihood_model.prms_def
                )
                residuals_list.append(residuals)
                # TODO: Doing this in the extension evaluation is very inefficient
                temp_extension_coordinates = np.array([])
                for variable in self.extension_variables:
                    try:
                        temp_extension_coordinates = np.vstack(
                            (
                                temp_extension_coordinates,
                                self.problem.experiments[
                                    likelihood_model.experiment_name
                                ].sensor_data[variable],
                            )
                        )
                    except ValueError:
                        temp_extension_coordinates = self.problem.experiments[
                            likelihood_model.experiment_name
                        ].sensor_data[variable]
                    if len(temp_extension_coordinates.shape) == 1:
                        temp_extension_coordinates = temp_extension_coordinates.reshape(
                            1, -1
                        )
                extension_coordinates = np.hstack(
                    (extension_coordinates, temp_extension_coordinates)
                )
            # TODO: In future, bias should have its own model that allows for input/output definition
            #       For now, we assume that the bias is a GP that takes the extension variable as input
            if self.problem.bias_parameters["computational_gp"] == "forward_model":
                self.problem.bias_parameters["computational_gp"] = list(
                    self.problem.likelihood_models.values()
                )[0].forward_model.response
            if self.problem.bias_parameters["derivative"] == "forward_model":
                self.problem.bias_parameters["derivative"] = self.generate_derivative(
                    list(self.problem.likelihood_models.values())[
                        0
                    ].forward_model.derivative,
                    len(self.problem.likelihood_models.values()),
                )
            self.problem.bias_parameters["evaluation_point"] = theta
            bias = self.problem.bias_model_class(**self.problem.bias_parameters)
            bias.train(
                self.scale_coordinates(np.array(extension_coordinates).transpose()),
                np.concatenate(residuals_list) * self.scale_residuals,
            )

            ############# BLOCK FOR DEBUGGING #############
            # prediction, covariance = bias.predict(self.scale_coordinates(np.array(extension_coordinates).transpose())[126:168], return_cov=True)
            # prediction_scaled = prediction/self.scale_residuals
            # covariance_scaled = covariance/self.scale_residuals**2

            # plt.figure(figsize=[6.4,4.8])
            # plt.plot(np.concatenate(residuals_list)[126:168], label="Residuals")
            # plt.plot(prediction_scaled, label="Bias prediction")
            # plt.title( r"Fitted bias for E = {:.2f} GPa and $\Delta T$ = 4.6E-5 K".format(theta[0]/1E12))
            # plt.xlabel("Timesteps [-]")
            # plt.ylabel("Residuals [m]")
            # plt.legend()
            # plt.savefig("./figures/bridge/bias_ogp_fitting_analysis.png")
            # plt.savefig("./figures/bridge/bias_ogp_fitting_analysis.pdf")
            # plt.savefig("./figures/bridge/bias_ogp_fitting_analysis.eps", format="eps", dpi = 800)
            # plt.show()

            # log_cov_matrix = np.log(np.abs(covariance_scaled))
            # plt.imshow(log_cov_matrix, cmap='coolwarm')
            # plt.colorbar()
            # plt.title('Covariance Matrix')
            # plt.show()

            ########### END BLOCK FOR DEBUGGING ###########

            # Save bias
            self.problem.bias_model = bias.clone_with_theta(theta)
            # Return log_marginal_likelihood of the bias GP
            # TODO: This should be more flexible in the future

            return float(bias.gp.log_marginal_likelihood())

        # Formulation for standard problem
        else:
            # compute the contribution to the log-likelihood function for each likelihood
            # model and sum it all up
            ll = 0.0
            for likelihood_model in self.problem.likelihood_models.values():
                # compute the model response/residuals for the likelihood model's experiment
                response, residuals = self.evaluate_model_response(
                    theta,
                    likelihood_model.forward_model,
                    likelihood_model.experiment_name,
                )
                # get the parameter values for the likelihood model's parameters
                prms_likelihood = self.problem.get_parameters(
                    theta, likelihood_model.prms_def
                )
                # evaluate the loglike-contribution for the likelihood model
                ll += likelihood_model.loglike(response, residuals, prms_likelihood)
            return ll

    def generate_derivative(self, derivative, number_likelihood_models):
        def derivative_wrapper(evaluation_point):
            derivative_list = []
            for i in range(number_likelihood_models):
                derivative_list.append(derivative(evaluation_point))
                # FIXME: This does not work if the derivatives are more than 1D
            return np.array(derivative_list).flatten() * self.scale_residuals

        return derivative_wrapper
    
class EmbeddedMCISolver(EmceeSolver):

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
        model_response_vector = vectorize_nd_numpy_dict(model_response_dict)
        mean_response_vector = np.mean(model_response_vector, axis=0)
        var_response_vector = np.var(model_response_vector, axis=0)
        model_response_vector = np.array([mean_response_vector, var_response_vector])

        # compute the residuals by comparing to the experimental response
        exp_response_dict = forward_model.output_from_experiments[experiment_name]
        # Reorder exmperiment response dict to match model response dict
        if not list(model_response_dict.keys()) == list(exp_response_dict.keys()):
            exp_response_dict = {
                key: exp_response_dict[key] for key in model_response_dict.keys()
            }
            forward_model.output_from_experiments[experiment_name] = exp_response_dict
        exp_response_vector = vectorize_numpy_dict(exp_response_dict)
        residuals_vector = exp_response_vector - mean_response_vector

        return model_response_vector, residuals_vector
    
class EmbeddedPCESolver(EmceeSolver):

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
        inp["experiment_name"] = experiment_name # Required for surrogate selection
        
        # evaluate the forward model and translate the result to a single vector
        model_response_dict = forward_model(inp)
        model_response_vector, dist = vectorize_tuple_pce_dict(model_response_dict)
        mean_response_vector = np.array([chaospy.E(response, dist) for response in model_response_vector]).flatten()
        var_response_vector = np.array([chaospy.Var(response, dist) for response in model_response_vector]).flatten()
        model_response_vector = np.array([mean_response_vector, var_response_vector])

        # compute the residuals by comparing to the experimental response
        exp_response_dict = forward_model.output_from_experiments[experiment_name]
        # Reorder exmperiment response dict to match model response dict
        if not list(model_response_dict.keys()) == list(exp_response_dict.keys()):
            exp_response_dict = {
                key: exp_response_dict[key] for key in model_response_dict.keys()
            }
            forward_model.output_from_experiments[experiment_name] = exp_response_dict
        exp_response_vector = vectorize_numpy_dict(exp_response_dict)
        residuals_vector = exp_response_vector - mean_response_vector

        return model_response_vector, residuals_vector
    
    def run(
        self,
        n_walkers: int = 20,
        n_steps: int = 1000,
        n_initial_steps: int = 100,
        true_values: Optional[dict] = None,
        **kwargs,
    ) -> az.data.inference_data.InferenceData:
        """
        Runs the emcee-sampler for the InverseProblem the EmceeSolver was initialized
        with and returns the results as an arviz InferenceData obj.

        Parameters
        ----------
        n_walkers
            Number of walkers used by the estimator.
        n_steps
            Number of steps to run.
        n_initial_steps
            Number of steps for initial (burn-in) sampling.
        true_values
            True parameter values, if known.
        kwargs
            Additional key-word arguments channeled to emcee.EnsembleSampler.

        Returns
        -------
        inference_data
            Contains the results of the sampling procedure.
        """

        # log which solver is used
        logger.info(
            f"Solving problem using emcee sampler with {n_initial_steps} + {n_steps} "
            f"samples and {n_walkers} walkers"
        )
        if kwargs:
            logger.info("Additional options:")
            print_dict_in_rows(kwargs, printer=logger.info)
        else:
            logger.info("No additional options specified")

        # draw initial samples from the parameter's priors
        logger.debug("Drawing initial samples")
        if self.seed is not None:
            np.random.seed(self.seed)
        sampling_initial_positions = np.zeros(
            (n_walkers, self.problem.n_latent_prms_dim)
        )
        theta_names = self.problem.get_theta_names(tex=False, components=False)
        for parameter_name in theta_names:
            idx = self.problem.parameters[parameter_name].index
            idx_end = self.problem.parameters[parameter_name].index_end
            samples = self.sample_from_prior(parameter_name, n_walkers)
            if (idx_end - idx) == 1:
                sampling_initial_positions[:, idx] = samples
            else:
                sampling_initial_positions[:, idx:idx_end] = samples

        # The following code is based on taralli and merely adjusted to the variables
        # in the probeye setup; see https://gitlab.com/tno-bim/taralli

        # ............................................................................ #
        #                                 Pre-process                                  #
        # ............................................................................ #

        def logprob(x):
            # Skip loglikelihood evaluation if logprior is equal
            # to negative infinity
            logprior = self.logprior(x)
            if logprior == -np.inf:
                return logprior

            # Otherwise return logprior + loglikelihood
            return logprior + self.loglike(x)

        logger.debug("Setting up EnsembleSampler")
        self.sampler = emcee.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=self.problem.n_latent_prms_dim,
            log_prob_fn=logprob,
            **kwargs,
        )

        if self.seed is not None:
            random.seed(self.seed)
            self.sampler.random_state = np.random.mtrand.RandomState(self.seed)

        # ............................................................................ #
        #        Initial sampling, burn-in: used to avoid a poor starting point        #
        # ............................................................................ #

        logger.debug("Starting sampling (initial + main)")
        start = time.time()
        state = self.sampler.run_mcmc(
            initial_state=sampling_initial_positions,
            nsteps=n_initial_steps,
            progress=self.show_progress,
        )
        self.sampler.reset()

        # ............................................................................ #
        #                          Sampling of the posterior                           #
        # ............................................................................ #
        self.sampler.run_mcmc(
            initial_state=state, nsteps=n_steps, progress=self.show_progress
        )
        end = time.time()
        runtime_str = pretty_time_delta(end - start)
        logger.info(
            f"Sampling of the posterior distribution completed: {n_steps} steps and "
            f"{n_walkers} walkers."
        )
        logger.info(f"Total run-time (including initial sampling): {runtime_str}.")
        logger.info("")
        logger.info("Summary of sampling results (emcee)")
        posterior_samples = self.sampler.get_chain(flat=True)
        with contextlib.redirect_stdout(stream_to_logger("INFO")):  # type: ignore
            self.summary = self.emcee_summary(
                posterior_samples, true_values=true_values
            )
        logger.info("")  # empty line for visual buffer
        self.raw_results = self.sampler

        # translate the results to a common data structure and return it
        self.var_names = self.problem.get_theta_names(tex=True, components=True)
        inference_data = az.from_emcee(self.sampler, var_names=self.var_names)
        return inference_data

    def restart_run(self, state, n_steps):
        self.sampler.run_mcmc(
            initial_state=state, nsteps=n_steps, progress=False)