# standard library imports
from typing import TYPE_CHECKING, Optional, Union, Callable

# third party imports
import numpy as np
import arviz as az
from loguru import logger
import matplotlib.pyplot as plt

# local imports
from probeye.subroutines import check_for_uninformative_priors
from probeye.inference.emcee.solver import EmceeSolver
from probeye.inference.koh.likelihood_models import translate_likelihood_model

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
        scale_residuals: float = 1.0
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
                raise Exception("Extension variable must be specified if extended problem is used.")
            else:
                self.extension_variables = extension_variables

            self.scale_coordinates = self.scale_coordinates if scale_coordinates_flag else lambda x: x

            # The extended model needs the bias defined in the inverse problem
            if not hasattr(problem, "bias_model_class") and not hasattr(problem, "bias_parameters"):
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
            extension_coordinates =np.array([]).reshape(len(self.extension_variables),0)
            ll = 0.0
            for likelihood_model in self.problem.likelihood_models.values():
                # compute the model response/residuals for the likelihood model's experiment
                response, residuals = self.evaluate_model_response(
                    theta, likelihood_model.forward_model, likelihood_model.experiment_name
                )
                prms_likelihood = self.problem.get_parameters(
                    theta, likelihood_model.prms_def
                )
                residuals_list.append(residuals)
                #TODO: Doing this in the extension evaluation is very inefficient
                temp_extension_coordinates = np.array([])
                for variable in self.extension_variables:
                    try:
                        temp_extension_coordinates = np.vstack((temp_extension_coordinates, self.problem.experiments[likelihood_model.experiment_name].sensor_data[variable]))
                    except ValueError:
                        temp_extension_coordinates = self.problem.experiments[likelihood_model.experiment_name].sensor_data[variable]
                if len(temp_extension_coordinates.shape) == 1:
                    temp_extension_coordinates = temp_extension_coordinates.reshape(1,-1)
                extension_coordinates = np.hstack((extension_coordinates, temp_extension_coordinates))
            # TODO: In future, bias should have its own model that allows for input/output definition
            #       For now, we assume that the bias is a GP that takes the extension variable as input
            bias = self.problem.bias_model_class(**self.problem.bias_parameters)
            bias.train(self.scale_coordinates(np.array(extension_coordinates).transpose()), np.concatenate(residuals_list)*self.scale_residuals)

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
                    theta, likelihood_model.forward_model, likelihood_model.experiment_name
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

        assert (self.problem.bias_parameters["computational_gp"] == "forward_model") or isinstance(self.problem.bias_parameters["computational_gp"], Callable), "Computational GP must be either a callable or 'forward_model'"
        assert (self.problem.bias_parameters["derivative"] == "forward_model") or isinstance(self.problem.bias_parameters["derivative"], Callable), "Derivative must be either a callable or 'forward_model'"
        # check whether the values of the latent parameters are within their domains;
        # they can end up outside their domains for example during sampling, when a
        # parameter vector is proposed, that contains a value that is not within the
        # specified bounds of a parameter

        if not self.problem.check_parameter_domains(theta):
            return -np.inf
        
        # Formulation for extended problem
        if self.extended_problem:
            residuals_list = []
            extension_coordinates =np.array([]).reshape(len(self.extension_variables),0)
            ll = 0.0
            for likelihood_model in self.problem.likelihood_models.values():
                # compute the model response/residuals for the likelihood model's experiment
                response, residuals = self.evaluate_model_response(
                    theta, likelihood_model.forward_model, likelihood_model.experiment_name
                )
                prms_likelihood = self.problem.get_parameters(
                    theta, likelihood_model.prms_def
                )
                residuals_list.append(residuals)
                #TODO: Doing this in the extension evaluation is very inefficient
                temp_extension_coordinates = np.array([])
                for variable in self.extension_variables:
                    try:
                        temp_extension_coordinates = np.vstack((temp_extension_coordinates, self.problem.experiments[likelihood_model.experiment_name].sensor_data[variable]))
                    except ValueError:
                        temp_extension_coordinates = self.problem.experiments[likelihood_model.experiment_name].sensor_data[variable]
                extension_coordinates = np.hstack((extension_coordinates, temp_extension_coordinates))
            # TODO: In future, bias should have its own model that allows for input/output definition
            #       For now, we assume that the bias is a GP that takes the extension variable as input
            if self.problem.bias_parameters["computational_gp"] == "forward_model":
                self.problem.bias_parameters["computational_gp"] = list(self.problem.likelihood_models.values())[0].forward_model.response
            if self.problem.bias_parameters["derivative"] == "forward_model":
                self.problem.bias_parameters["derivative"] = self.generate_derivative(list(self.problem.likelihood_models.values())[0].forward_model.derivative, len(self.problem.likelihood_models.values()))
            self.problem.bias_parameters["evaluation_point"] = theta
            bias = self.problem.bias_model_class(**self.problem.bias_parameters)
            bias.train(self.scale_coordinates(np.array(extension_coordinates).transpose()), np.concatenate(residuals_list)*self.scale_residuals)

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
                    theta, likelihood_model.forward_model, likelihood_model.experiment_name
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
                #FIXME: This does not work if the derivatives are more than 1D
            return np.array(derivative_list).flatten()*self.scale_residuals
        return derivative_wrapper
