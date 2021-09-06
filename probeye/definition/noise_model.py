# third party imports
import numpy as np

# local imports
from probeye.subroutines import len_or_one
from probeye.subroutines import list2dict, make_list


class NoiseModelTemplate:
    def __init__(self, prms_def, sensors):
        """
        Parameters
        ----------
        prms_def : str, list[str], dict
            A list of parameter names (strings) defining how a noise parameter
            vector given to the loglike_contribution method is interpreted. E.g.
            prms_def = ['mu', 'sigma'] means that the noise parameter vector
            has two elements, the first of which gives the value of 'mu' and the
            second gives the value of 'sigma'.
        sensors : str, list[str]
            Names of the sensors that are required to evaluate the noise model.
        """
        self.prms_def = list2dict(make_list(prms_def))
        self.prms_dim = len_or_one(make_list(prms_def))
        self.sensors = make_list(sensors)
        # this is a list of experiment names, that relate to the noise model;
        # the list will be filled after experiments have been added to the
        # InferenceProblem and the problem definition is complete; in this case
        # call InferenceProblem.assign_experiments_to_noise_models() to fill
        # this list with the corresponding names
        self.experiment_names = []
        # as soon as defined, this attribute will be a pointer to the inference
        # problems experiments (it will be used for consistency checks)
        self.problem_experiments = {}

    def add_experiment_names(self, experiment_names_):
        """
        Adds experiment names to the noise model. When the noise model is
        evaluated it will only be evaluated for those experiments added here.

        Parameters
        ----------
        experiment_names_ : str, list[str]
            Names (strings) of experiments from the InferenceProblem that should
            be added to the noise model.
        """
        # check if the given experiments are compatible with the noise model
        # with respect to the sensors
        experiment_names = make_list(experiment_names_)
        forward_models = set()
        for exp_name in experiment_names:
            exp_dict = self.problem_experiments[exp_name]
            forward_models.add(exp_dict['forward_model'])
            sensor_names_exp = [*exp_dict['sensor_values'].keys()]
            for sensor_name in self.sensors:
                if sensor_name not in sensor_names_exp:
                    raise RuntimeError(
                        f"Experiment '{exp_name}' does not contain a sensor "
                        f"'{sensor_name}' which is required for the evaluation "
                        f"of the noise model.")
        # check if the given experiments all refer to one forward model
        if len(forward_models) > 1:
            raise RuntimeError(
                f"The given experiments refer to more than one forward model!")
        # check if one of the given experiments have been added before
        for exp_name in experiment_names:
            if exp_name in self.experiment_names:
                raise RuntimeError(
                    f"The experiment '{exp_name}' has already been added to "
                    f"this noise model. Something might be wrong here.")
        self.experiment_names += experiment_names

    def error(self, model_response_dict):
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
            1D numpy arrays representing the model errors as values.
        """
        # prepare the dictionary keys
        model_error_dict = {name: np.array([]) for name in self.sensors}

        # fill the dictionary with model error vectors
        for exp_name in self.experiment_names:
            exp_dict = self.problem_experiments[exp_name]
            ym_dict = model_response_dict[exp_name]
            ye_dict = exp_dict['sensor_values']
            me_dict = self.error_function(ym_dict, ye_dict)
            model_error_dict =\
                {name: np.append(model_error_dict[name], me_dict[name])
                 for name in self.sensors}

        return model_error_dict

    def error_function(self, ym_dict, ye_dict):
        """
        Evaluates the model error for each of the noise model's sensors.

        Parameters
        ----------
        ym_dict : dict
            The computed values for the model's output sensor_values.
        ye_dict : dict
            The measured values for the model's output sensor_values.

        Returns
        -------
        error_dict : dict
            The computed model error for the model's output sensor_values.
        """
        raise NotImplementedError(
            "Your model does not have a error_function-method yet. If you " +
            "want to use it, you need to implement it first.")

    def loglike_contribution(self, model_response_dict, prms):
        """
        Evaluates the log-likelihood function for the given model error and
        the given noise parameter vector. This method has to be overwritten.

        Parameters
        ----------
        model_response_dict : dict
            The first key is the name of the experiment. The values are dicts
            which contain the forward model's output sensor's names as keys
            have the corresponding model responses as values.
        prms : ParameterList-object
            Dictionary-like object containing parameter name:value pairs.

        Returns
        -------
        ll : float
            The evaluated log-likelihood function.
        """
        raise NotImplementedError(
            "Your model does not have a loglike_contribution-method. You " +
            "need to define this method so you can evaluate your noise model.")
