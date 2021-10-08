# third party imports
import torch as th
import pyro
import pyro.distributions as dist

# local imports
from probeye.definition.noise_model import NormalNoiseModel


class NormalNoise(NormalNoiseModel):
    """
    A general Gaussian (normal) noise model with or without correlations. Note
    that currently, there are no correlation capabilities defined.
    """
    def __init__(self, target_sensor, prms_def, name=None, corr=None,
                 corr_model='exp', noise_type='additive', use_gradients=True):
        """
        For information on most of the above arguments check out the docstring
        of the parent class's __init__ method.

        Parameters
        ----------
        use_gradients : bool, optional
            When True, the noise model will be defined to be compatible with
            torch.Tensors. More precisely, the error-method of this class will
            be defined to expect to work with torch.Tensors. When False, the
            default error-method of the parent class will be used which does
            work with numpy-arrays but not with torch.Tensors.
        """
        # initialize the super-class (NormalNoiseModel) based on the given input
        super().__init__(prms_def=prms_def, sensors=target_sensor, name=name,
                         corr=corr, corr_model=corr_model,
                         noise_type=noise_type)

        # the following attributes are not considered in the parent class
        self.target_sensor = target_sensor
        self.use_gradients = use_gradients

        # the error-method is defined based on whether to use gradients or not
        if self.use_gradients:
            self.error = self.error_torch

    def error_torch(self, model_response_dict):
        """
        Computes the model error for all of the noise model's experiments and
        returns them in a dictionary that is sorted by output sensor_values.

        Parameters
        ----------
        model_response_dict : dict
            The first key is the name of the experiment. The values are dicts
            which contain the forward model's output sensor's names as keys
            have the corresponding model responses as values.

        Returns
        -------
        model_error : dict
            A dictionary with the keys being the noise model's sensor names, and
            torch.Tensors representing the model errors as values.
        """

        # prepare the dictionary keys
        model_error_dict = {name: th.Tensor([]) for name in self.sensor_names}

        # fill the dictionary with model error vectors
        for exp_name in self.experiment_names:
            exp_dict = self.problem_experiments[exp_name]
            ym_dict = model_response_dict[exp_name]
            ye_dict = exp_dict['sensor_values']
            me_dict = self.error_function(ym_dict, ye_dict)
            model_error_dict =\
                {name: th.cat((model_error_dict[name], me_dict[name]))
                 for name in self.sensor_names}
        return model_error_dict

    def sample_cond_likelihood(self, model_response, prms, use_gradients):
        """
        Creates a likelihood-sample conditioned on the observed errors.

        model_response_dict : dict
            The first key is the name of the experiment. The values are dicts
            which contain the forward model's output sensor's names as keys
            have the corresponding model responses as values.
        prms : ParameterList-object
            Dictionary-like object containing parameter name:value pairs.
        use_gradients : bool
            Flag from probeye's pyro-solver that effectively states, if the
            latent parameters have to be detached from the graph or not.
        """
        std = prms['std']
        mean = 0.0 if self.zero_mean else prms['mean']
        # compute the model error; note that this mode has exactly one sensor
        model_error_vector = self.error(model_response)[self.target_sensor.name]
        if not use_gradients:
            model_error_vector = th.tensor(model_error_vector)
        pyro.sample(f'lkl_{self.name}', dist.Normal(mean, std),
                    obs=model_error_vector)

def translate_noise_model(noise_base, use_gradients=True):
    """
    Translates a given instance of NoiseBase (which is essentially just a
    description of the noise model without computing-methods) to a specific
    noise model object which does contain computing-methods (e.g. compute the
    log-likelihood contribution).

    Parameters
    ----------
    noise_base : obj[NoiseBase]
        An instance of NoiseBase which contains basic information on the noise
        model but no computing-methods.
    use_gradients : bool, optional
        When True, torch's autograd functionality is used. In this case, the
        provided model has to accept tensor-valued input. If this value is set
        to False, the latent-parameter vector will be detached from torch's
        computational graph before evaluating the likelihood, which makes the
        sampling procedure extremely slow.

    Returns
    -------
    noise_object : obj[NoiseBase]
        An instance of a specific noise model class with computing capabilities.
        Examples for such classes are given above in this file.
    """

    # this is the noise classes currently defined (see code above)
    noise_classes = {'normal': NormalNoise}

    # this is where the translation happens
    noise_object = noise_classes[noise_base.dist](
        target_sensor=noise_base.sensors[0], prms_def=noise_base.prms_def,
        name=noise_base.name, corr=noise_base.corr,
        corr_model=noise_base.corr_model, noise_type=noise_base.noise_type,
        use_gradients=use_gradients)

    # here, we take the assigned experiments from the base object
    noise_object.experiment_names = noise_base.experiment_names
    noise_object.problem_experiments = noise_base.problem_experiments

    return noise_object
