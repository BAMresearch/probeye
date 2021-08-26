# standard library imports
import copy as cp

# third party imports
import numpy as np

# local imports
from probeye.subroutines import delta_x
from probeye.subroutines import len_or_one
from probeye.subroutines import list2dict


class InputSensor:
    """
    Base class for an input sensor of the forward model. Essentially just a
    class with a 'name' attribute.
    """
    def __init__(self, name):
        """
        Parameters
        ----------
        name : string
            The name of the sensor, e.g. 'Time'.
        """
        self.name = name


class OutputSensor:
    """
    Base class for an output sensor of the forward model. Each model response
    of the forward model needs to be associated with an output sensor. At least
    such an object must have a name, but can also have additional attributes
    such as its position or its temperature. In these cases the user has to
    define his own output sensor class, which should be derived from this one.
    """
    def __init__(self, name):
        """
        Parameters
        ----------
        name : string
            The name of the sensor, e.g. 'Deflection-Sensor bottom-left'.
        """
        self.name = name


class ModelTemplate:
    """
    This class serves as a base class for any forward model. When you want to
    define a specific forward model, you need to derive your own class from this
    one, and then define the 'response' method. The latter essentially describes
    the model function mapping the model input to the output.
    """
    def __init__(self, prms_def_, input_sensors, output_sensors):
        """
        Parameters
        ----------
        prms_def_ : list
            Contains the model's calibration parameter names. The list may only
            contain strings or one-element dictionaries. It could look, for
            example, like [{'a': 'm'}, 'b']. The one-element dictionaries
            account for the possibility to define a local name for a calibration
            parameter that is different from the global name. In the example
            above, the calibration parameter with the global name 'a' will be
            referred to as 'm' within the model. So, the one-element dicts have
            the meaning {<global name>: <local name>}. String-elements are
            interpreted as having similar local and global names. Note that the
            local-name option will not be required most of the times.
        input_sensors : list[InputSensor]
            Contains sensor-objects structuring the model input.
        output_sensors : list[OutputSensor]
            Contains sensor-objects structuring the model output.
        """

        # convert the given parameter names to a dictionary with global names
        # as keys and local names as values
        self.prms_def = list2dict(prms_def_)

        # other attributes
        self.prms_dim = len_or_one(self.prms_def)
        self.input_sensors = input_sensors
        self.output_sensors = output_sensors

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
            "Your model does not have a proper response-method yet. You need " +
            "to define this method, so you can evaluate your model."
        )

    def jacobian(self, exp_inp, prms):
        """
        Computes the gradient of the response function at exp_inp and prms with
        respect to the calibration parameters prms. A symmetric difference
        scheme is used here.

        Parameters
        ----------
        exp_inp : dict
            The experimental input data of the model. The keys are names of the
            experiment's input sensors, and the values are their numeric values.
        prms : dict
            The calibration parameters with their local names as keys and their
            numeric values as values.

        Returns
        -------
        jac : dict
            Contains the calibration parameter names as keys. The values are
            dictionaries with the model's output sensor names as keys and the
            respective approximated derivatives as values.
        """
        jac = {}
        for prm_name, prm_value in prms.items():
            # evaluate the model at prms_i - h
            h = delta_x(prm_value)
            prms_left = cp.copy(prms)
            prms_left[prm_name] = prm_value - h
            left = self({**exp_inp, **prms_left})
            # evaluate the model at prms_i + h
            prms_right = cp.copy(prms)
            prms_right[prm_name] = prm_value + h
            right = self({**exp_inp, **prms_right})
            # evaluate the symmetric difference scheme
            jac[prm_name] = {}
            for sensor_name in left.keys():
                jac[prm_name][sensor_name] =\
                    (right[sensor_name] - left[sensor_name]) / (2 * h)
        return jac

    def error_function(self, ym_dict, ye_dict):
        """
        Evaluates the model error (a metric for the deviation of the forward
        model's computed result with respect to the measured data from the
        experiments) for one experiment. The default metric reflected below is a
        simple sensor-by-sensor computation of an additive model error. This
        function should be overwritten if another definition of the model error
        should be applied.

        Parameters
        ----------
        ym_dict : dict
            The computed values for the model's output sensors.
        ye_dict : dict
            The measured values for the model's output sensors.

        Returns
        -------
        error_dict : dict
            The computed model error for the model's output sensors.
        """
        # for each sensor, its own error metric is used to compute the error
        error_dict = {os.name: ym_dict[os.name] - ye_dict[os.name]
                      for os in self.output_sensors}
        return error_dict

    def error(self, prms, experiments):
        """
        Computes the model error for all given experiments and returns them in
        a dictionary that is sorted by output sensors.

        Parameters
        ----------
        prms : dict
            The calibration parameters with their local names as keys and their
            numeric values as values.
        experiments : dict
            Contains the experiment names (strings) as keys, while the values
            are dicts with the structure <sensor name>: <sensor value>. Note
            that <sensor value> might be a vector.

        Returns
        -------
        model_error : dict
            A dictionary with the keys being the output sensor names, and lists
            of numbers representing the model errors as values.
        """
        # prepare the dictionary keys
        model_error_dict = {os.name: np.array([]) for os in self.output_sensors}

        # fill the dictionary with model error vectors
        for exp_dict in experiments.values():
            # prepare the model input values from the experimental data
            exp_inp = {input_sensor.name: exp_dict[input_sensor.name]
                       for input_sensor in self.input_sensors}
            inp = {**exp_inp, **prms}  # adds the two dictionaries
            ym_dict = self(inp)
            ye_dict = {output_sensor.name: exp_dict[output_sensor.name]
                       for output_sensor in self.output_sensors}
            me_dict = self.error_function(ym_dict, ye_dict)
            model_error_dict =\
                {os.name: np.append(model_error_dict[os.name], me_dict[os.name])
                 for os in self.output_sensors}

        return model_error_dict
