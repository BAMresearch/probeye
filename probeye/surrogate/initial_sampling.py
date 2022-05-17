# standard library
from typing import Tuple
import copy

# third party imports
import numpy as np
import pandas as pd
from scipy.stats import qmc

# local imports
from probeye.definition.inverse_problem import InverseProblem
from probeye.inference.priors import translate_prior
from probeye.subroutines import len_or_one


class LatinHypercubeSampler:
    """
    Contains functionalities to provide samples of an inverse problem's parameters by
    utilizing latin hypercube sampling. The generate samples are intended for training
    of a surrogate model.

    Parameters
    ----------
    problem
        The considered inverse problem.
    """

    def __init__(self, problem: InverseProblem):

        # the considered inverse problem
        self.problem = problem

        # the sampling happens before the problem is given to a solver; this means that
        # the priors of the problem are merely descriptive and they have to be
        # translated to have their respective computing routines
        self.priors = copy.deepcopy(self.problem.priors)
        for prior_template in self.problem.priors.values():
            prm_name = prior_template.ref_prm
            self.priors[prm_name] = translate_prior(prior_template)

    def generate_samples(self, n_samples: int, seed: int = 1) -> np.ndarray:
        """
        Samples the problems latent parameter vector from the parameter's priors in
        combination with latin hypercube sampling.

        Parameters
        ----------
        n_samples
            The number of requested samples.
        seed
            Seed for random number generator.

        Returns
        -------
        sample_array
            The sampled latent parameter vectors. Each row corresponds to a single
            latent parameter vector.
        """

        # make sure that all parameters are one-dimensional; it is not straight forward
        # how to do LHS for general multivariate parameters
        for prm_name in self.problem.latent_prms:
            if self.problem.parameters[prm_name].dim > 1:
                raise RuntimeError(
                    f"The given problem has a multivariate parameter ('{prm_name}') "
                    f"for which LHS is not supported."
                )

        # the dimension of the required hypercube is the number of parameters; note
        # that this only holds since all parameters are 1D (see the check above)
        dim = self.problem.n_latent_prms

        # create the requested number of latin hypercube samples with the requested dim.
        lhd = qmc.LatinHypercube(d=dim, seed=seed).random(n=n_samples)

        # this is going to be the array of the parameter samples; each row will
        # correspond to a theta-vector
        sample_array = np.zeros(lhd.shape)

        # fill the prepared array for the parameter samples using the LHS samples
        for prm_name, parameter in self.problem.parameters.items():
            if parameter.is_latent:
                idx = parameter.index
                prior = self.priors[prm_name]
                prms = self.problem.get_constants(prior.hyperparameters)
                for lt_prm in self.problem.get_latent_prior_hyperparameters(prm_name):
                    idx_lt = self.problem.parameters[lt_prm].index
                    prms[lt_prm] = sample_array[:, idx_lt]
                q = lhd[:, idx]
                sample_array[:, idx] = prior(prms, "ppf", q, use_ref_prm=False)

        return sample_array

    def generate_training_data(
        self,
        forward_model_name: str,
        n_samples: int,
        seed: int = 1,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Generates a given number of training data for fitting a surrogate model. The
        training data contains a number of parameter vectors (sampled using LHS) and
        the corresponding model responses.

        Parameters
        ----------
        forward_model_name
            The name of the forward model that should be evaluated.
        n_samples
            The number of parameter vectors the forward model should be evaluated for.
        seed
            Seed for random number generator.

        Returns
        -------
        prm_samples_pd
            The parameter samples the forward model was evaluated at.
        responses_over_experiments
            The keys are the names of the experiment the forward model is associated
            with, while the values are 3D-arrays containing the forward model's
            response. responses_over_experiments[i][j] will contain the forward  model's
            response with the ith parameter-vector for the jth output sensor.
        """

        # get the forward model object with the given name and prepare the corresponding
        # experimental in- and output dictionaries
        forward_model = self.problem.forward_models[forward_model_name]
        forward_model.prepare_experimental_inputs_and_outputs()

        # generate the latent parameter samples and convert it to a data frame to have
        # the association between columns and parameter names
        prm_samples = self.generate_samples(n_samples, seed=seed)
        prm_samples_pd = pd.DataFrame(
            prm_samples, columns=self.problem.get_theta_names()
        )

        # this dictionary will contain the forward model responses for each of the
        # experiments associated with the forward model; so the keys will be experiment
        # names while the values will be 3D-arrays with the forward model's responses;
        # responses_over_experiments[i] will correspond to the response of the ith
        # parameter vector; responses_over_experiments[i][j] will contain the forward
        # model's response with the ith parameter vector for the jth output sensor
        responses_over_experiments = {}

        # here, the length of the vectors of the forward model's output sensors is
        # determined; to that end, the forward model is evaluated once
        first_exp_name = forward_model.experiment_names[0]
        exp_inp = forward_model.input_from_experiments[first_exp_name]
        first_theta = prm_samples[0]
        prms_model = self.problem.get_parameters(first_theta, forward_model.prms_def)
        inp = {**exp_inp, **prms_model}
        response_dict = forward_model.response(inp)
        # make sure that the vectors returned by each of the forward model's output
        # sensors has the same length; otherwise an AssertionError is raised
        length_set = set()
        for value in response_dict.values():
            length_set.add(len_or_one(value))
        assert len(length_set) == 1
        n_out_values = list(length_set)[0]

        # evaluate the forward model for each experiment/parameter vector
        for exp_name in forward_model.experiment_names:
            exp_inp = forward_model.input_from_experiments[exp_name]
            response_array = np.zeros(
                (n_samples, forward_model.n_output_sensors, n_out_values)
            )
            for i, theta in enumerate(prm_samples):
                prms_model = self.problem.get_parameters(theta, forward_model.prms_def)
                inp = {**exp_inp, **prms_model}  # adds the two dictionaries
                response_dict = forward_model.response(inp)
                for j, response_vector in enumerate(response_dict.values()):
                    response_array[i, j, :] = response_vector
            responses_over_experiments[exp_name] = response_array

        return prm_samples_pd, responses_over_experiments
