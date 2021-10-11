# local imports
from probeye.subroutines import translate_prms_def
from probeye.subroutines import make_list


class ForwardModelBase:
    """
    This class serves as a base class for any forward model. When you want to
    define a specific forward model, you need to derive your own class from this
    one, and then define the '__call__' method. The latter essentially describes
    the model function mapping the model input to the output.
    """
    def __init__(self, prms_def_, input_sensors, output_sensors):
        """
        Parameters
        ----------
        prms_def_ : str, list, dict
            Contains the model's latent parameter names. The list may only
            contain strings or one-element dictionaries. It could look, for
            example, like [{'a': 'm'}, 'b']. The one-element dictionaries
            account for the possibility to define a local name for a latent
            parameter that is different from the global name. In the example
            above, the latent parameter with the global name 'a' will be
            referred to as 'm' within the model. So, the one-element dicts have
            the meaning {<global name>: <local name>}. String-elements are
            interpreted as having similar local and global names. Note that the
            local-name option will not be required most of the times. The input
            from global to local name can also be provided as a dict. In the
            example above it would look like {'a': 'm', 'b': 'b'}.
        input_sensors : Sensor, list[Sensor]
            Contains sensor-objects structuring the model input.
        output_sensors : Sensor, list[Sensor]
            Contains sensor-objects structuring the model output.
        """

        # convert the given parameter names to a dictionary with global names
        # as keys and local names as values
        self.prms_def, self.prms_dim = translate_prms_def(prms_def_)

        # other attributes
        self.input_sensors = make_list(input_sensors)
        self.output_sensors = make_list(output_sensors)

    @property
    def input_sensor_names(self):
        """Provides input_sensor_names attribute."""
        return [sensor.name for sensor in self.input_sensors]

    @property
    def output_sensor_names(self):
        """Provides input_sensor_names attribute."""
        return [sensor.name for sensor in self.output_sensors]

    def __call__(self, inp):
        """
        Evaluates the model response and provides computed results for all of
        the model's output sensors. This method has to be overwritten by the
        user.

        Parameters
        ----------
        inp : dict
            Contains both the exp. input data and the  model's parameters. The
            keys are the names, and the values are their numeric values.

        Returns
        -------
        response_dict : dict
            Contains the model response (value) for each output sensor,
            referenced by the output sensor's name (key).
        """
        raise NotImplementedError(
            "Your model does not have a proper __call__-method yet. You need "
            "to define this method, so you can evaluate your model.")
