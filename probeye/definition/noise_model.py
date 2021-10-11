# third party imports
import numpy as np

# local imports
from probeye.subroutines import make_list, translate_prms_def


class NoiseModelBase:
    def __init__(self, dist, prms_def, sensors, name=None, corr=None,
                 corr_model=None, noise_type='additive'):
        """
        Parameters
        ----------
        dist : str
            A string specifying the probability distribution the noise model
            is based on, e.g. 'normal'.
        prms_def : str, list[str], dict
            A list of parameter names (strings) defining how a noise parameter
            vector given to the loglike_contribution method is interpreted. E.g.
            prms_def = ['mu', 'sigma'] means that the noise parameter vector
            has two elements, the first of which gives the value of 'mu' and the
            second gives the value of 'sigma'.
        sensors : obj[Sensor], list[Sensor]
            Sensor objects that are required to evaluate the noise model.
        name : str, None, optional
            Unique name of the noise model. This name is None, if the user does
            not specify it when adding the noise model to the problem. It is
            then named automatically before starting the inference engine.
        corr : dict, None, optional
            Defines the correlation model. So far this is just a placeholder.
            It is not clear yet how exactly the correlation should be defined.
            When it is set to None, all sensors/sensor elements are independent.
        corr_model : str, optional
            Defines the correlation function to be used in case corr isn't None.
        noise_type : str, optional
            Either 'additive', 'multiplicative' or 'other'. Defines if the error
            is computed via [prediction - measurement] ('additive') or via
            [prediction/measurement-1] ('multiplicative') or in some 'other'
            i.e., non-standard fashion.
        """
        self.dist = dist
        self.prms_def, self.prms_dim = translate_prms_def(prms_def)
        self.sensors = make_list(sensors)
        self.sensor_names = [sensor.name for sensor in self.sensors]
        self.name = name
        self.corr = corr
        self.corr_model = corr_model
        self.noise_type = noise_type
        # this is a list of experiment names, that relate to the noise model;
        # the list will be filled after experiments have been added to the
        # InferenceProblem and the problem definition is complete; in this case
        # call InferenceProblem.assign_experiments_to_noise_models() to fill
        # this list with the corresponding names
        self.experiment_names = []
        # as soon as defined, this attribute will be a pointer to the inference
        # problems experiments (it will be used for consistency checks)
        self.problem_experiments = {}

        # set the error_function depending on the noise-type
        if noise_type == "additive":
            self.error_function = self.error_function_additive
        elif noise_type == "multiplicative":
            self.error_function = self.error_function_multiplicative
        elif noise_type == "other":
            self.error_function = self.error_function_other
        else:
            raise ValueError(
                f"Encountered unknown noise_type: '{noise_type}'. The noise"
                f"_type must be either 'additive', 'multiplicative' or 'other'."
            )

    def add_experiments(self, experiment_names_):
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
            for sensor_name in self.sensor_names:
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
        model_error_dict = {name: np.array([]) for name in self.sensor_names}

        # fill the dictionary with model error vectors
        for exp_name in self.experiment_names:
            exp_dict = self.problem_experiments[exp_name]
            ym_dict = model_response_dict[exp_name]
            ye_dict = exp_dict['sensor_values']
            me_dict = self.error_function(ym_dict, ye_dict)
            model_error_dict =\
                {name: np.append(model_error_dict[name], me_dict[name])
                 for name in self.sensor_names}

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

    def error_function_additive(self, ym_dict, ye_dict):
        """
        Evaluates the additive model error for each of the noise model' sensors.

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
        # for each sensor, its own error metric is used to compute the error
        error_dict = {name: ym_dict[name] - ye_dict[name]
                      for name in self.sensor_names}
        return error_dict

    def error_function_multiplicative(self, ym_dict, ye_dict):
        """
        Evaluates the multiplicative model error for each of the noise model's
        sensors.

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
        # for each sensor, its own error metric is used to compute the error
        error_dict = {name: ym_dict[name] / ye_dict[name] - 1.0
                      for name in self.sensor_names}
        return error_dict

    def error_function_other(self, ym_dict, ye_dict):
        """
        Non-standard error function self.error_function will point to when
        self.noise_type is set to 'other'. See self.error_function for more
        information.
        """
        raise NotImplementedError(
            "Your model does not have an non-standard error_function-method "
            "yet. If you want to use it, you need to implement it first.")

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


class NormalNoiseModel(NoiseModelBase):
    """
    A general Gaussian (normal) noise model with or without correlations.
    """

    def __init__(self, prms_def, sensors, name=None, corr=None, corr_model=None,
                 noise_type='additive'):
        """
        See docstring of NoiseModelBase for information on the input arguments.
        """

        # initialize the base class with given input
        super().__init__('normal', prms_def, sensors, name=name, corr=corr,
                         corr_model=corr_model, noise_type=noise_type)

        # check that at the standard deviation is provided (this can be either
        # as a constant or a latent parameter, but it has to be given)
        if 'std' not in [*self.prms_def.values()]:
            raise RuntimeError(
                "The standard deviation 'std' was not provided in prms_def!")

        # the mean value(s) do not have to be stated explicitly; if they are not
        # given, the are assumed to be zero
        self.zero_mean = True
        if 'mean' in [*self.prms_def.values()]:
            self.zero_mean = False
