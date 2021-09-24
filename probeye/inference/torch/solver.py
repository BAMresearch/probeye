import logging

# third party imports
import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import pyro
import pyro.distributions as dist
from numpy.core._multiarray_umath import ndarray
from pyro.infer import EmpiricalMarginal, Importance, NUTS, MCMC, HMC
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# local imports
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.sensor import Sensor
from probeye.definition.inference_problem import InferenceProblem
from probeye.definition.noise_model import NormalNoise


def Pyro_torch_solver(problem, step_size=0.1, num_samples=300, warmup_steps=100):
    """
    
    :param problem: obj[InferenceProblem]
        Describes the inference problem including e.g. parameters and data.
    :param step_size: The step size of the NUTS sampler
    :param num_samples: int, optional
        The number of steps the sampler takes
    :param warmup_steps: int, optional
        The burnin steps
    :return: obj[mcmc]
        Contains the results of the sampling procedure
    """
    sampled_params = {}

    def prior_samples_dict(problem):
        """
        Function to create hierarchical prior graph, sample it and store in a dict with key from ProbEye inference problem obj.
        :param problem:
        :return:
        """
        for key, value in problem.priors.items():  # loops over all the priors
            if value.prior_type == "normal":
                hyperparameters = []
                for hyperpara_name in value.prms_def_no_ref.keys():  # loops over the corresponding hyperparameters
                    if hyperpara_name in sampled_params: # passing samples of hyperprior
                        hyperparameters.append(sampled_params[hyperpara_name])
                    else:
                        hyperparameters.append(problem.parameters[hyperpara_name].value)

                sampled_params[value.ref_prm] = pyro.sample(value.ref_prm,
                                                            dist.Normal(hyperparameters[0], hyperparameters[1]))

            if value.prior_type == "uniform":
                hyperparameters = []
                for hyperpara_name in value.prms_def_no_ref.keys():
                    hyperparameters.append(problem.parameters[hyperpara_name].value)
                sampled_params[value.ref_prm] = pyro.sample(value.ref_prm,
                                                            dist.Uniform(hyperparameters[0], hyperparameters[1]))
        return sampled_params

    def likelihood(prior_samples):
        """
        TODO: Add pyro based noise models.

        :param prior_samples: dict
            Contains dict of samples of prior
        :return:
        """
        for noise_model in problem.noise_models:
            # compute the model response for the noise model's experiment_names
            theta = th.stack(list(prior_samples.values())).detach().numpy()
            model_response = problem.evaluate_model_response(theta)[noise_model.experiment_names[0]]

            exp_dict = noise_model.problem_experiments[noise_model.experiment_names[0]]
            # print(model_response)
            ye_dict = exp_dict['sensor_values']
            # print(ye_dict)
            noise_sd = list(noise_model.prms_def.keys())[0]
            pyro.sample(f'lkl_{noise_model.experiment_names[0]}',
                        dist.Normal(th.tensor(model_response[noise_model.sensors[0]]), prior_samples[noise_sd]),
                        obs=th.tensor(ye_dict[noise_model.sensors[0]]))

    def posterior_model():
        prior_samples = prior_samples_dict(problem)

        likelihood(prior_samples)

    kernel = NUTS(posterior_model, step_size=step_size)
    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc.run()
    mcmc.summary()

    return mcmc
