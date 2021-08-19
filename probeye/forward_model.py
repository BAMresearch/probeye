# standard library imports
import copy as cp

# third party imports
import numpy as np

# local imports
from probeye.jacobian import delta_x
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
            The name of the sensor, e.g. 'Deflection-Sensor bottom-left'.
        """
        self.name = name


class OutputSensor:
    """
    Base class for an output sensor of the forward model. Each model response
    of the forward model needs to be associated with an output sensor. At least
    such an object must have a name and an error_metric, but can also have
    additional attributes such as its position or its temperature. In these
    cases the user has to define his own output sensor class, which should be
    derived from this one.
    """
    def __init__(self, name, error_metric='abs'):
        """
        Parameters
        ----------
        name : string
            The name of the sensor, e.g. 'Deflection-Sensor bottom-left'.
        error_metric : string, optional
            Either 'abs' (absolute) or 'rel' (relative). Defines if the model
            error with respect to this sensor should be measured in absolute or
            relative terms. In the former case ('abs'), the error is defined as
            (model prediction - measured value). In the latter case ('rel') the
            error is defined as (1 - (model prediction / measured value)).
        """
        self.name = name
        self.error_metric = error_metric


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

    def __call__(self, exp_inp, prms):
        """
        Evaluates the model response for each output sensor and returns the
        response dictionary.

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
        response_dict : dict
            Contains the model response (value) for each output sensor,
            referenced by the output sensor's name (key).
        """
        inp = {**exp_inp, **prms}  # adds the two dictionaries
        response_dict = {output_sensor.name: self.response(inp, output_sensor)
                         for output_sensor in self.output_sensors}
        return response_dict

    def response(self, inp, sensor):
        """
        Evaluates the model for the given sensor. This method has to be
        overwritten by the user's response method.

        Parameters
        ----------
        inp : dict
            Contains both the exp. input data and the  model's parameters. The
            keys are the names, and the values are their numeric values.
        sensor : object
            The output sensor the response should be evaluated for.
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
            left = self(exp_inp, prms_left)
            # evaluate the model at prms_i + h
            prms_right = cp.copy(prms)
            prms_right[prm_name] = prm_value + h
            right = self(exp_inp, prms_right)
            # evaluate the symmetric difference scheme
            jac[prm_name] = {}
            for sensor_name in left.keys():
                jac[prm_name][sensor_name] =\
                    (right[sensor_name] - left[sensor_name]) / (2 * h)
        return jac

    def error_function(self, exp_inp, prms, output_sensor, ye):
        """
        Evaluates the model error for a single experiment. This function can be
        overwritten if another definition of the model error should be applied.

        Parameters
        ----------
        exp_inp : dict
            The experimental input data of the model. The keys are names of the
            model's input sensors, and the values are their numeric values
            measured in the experiment.
        prms : dict
            The calibration parameters with their local names as keys and their
            numeric values as values.
        output_sensor : obj[OutputSensor]
            One of the model's output sensors the error should be evaluated for.
        ye : float
            The experimental output value for the given output_sensor.

        Returns
        -------
        error : float
            The computed model error for the given output sensor.
        """

        # compute the model prediction for the given sensor
        ym = self(exp_inp, prms)[output_sensor.name]

        # compute the error according to the defined error metric
        if output_sensor.error_metric == 'abs':
            error = ym - ye
        elif output_sensor.error_metric == 'rel':
            error = 1.0 - ym / ye
        else:
            raise ValueError(
                f"Output sensor '{output_sensor.name}' has an unknown error "
                f"metric: '{output_sensor.error_metric}'."
            )

        return error

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
        model_error_dict = {output_sensor.name: np.array([])
                            for output_sensor in self.output_sensors}

        # fill the dictionary with model error vectors
        for output_sensor in self.output_sensors:
            for exp_dict in experiments.values():
                # prepare the model input values from the experimental data
                exp_inp = {input_sensor.name: exp_dict[input_sensor.name]
                           for input_sensor in self.input_sensors}
                ye = exp_dict[output_sensor.name]
                me = self.error_function(exp_inp, prms, output_sensor, ye)
                model_error_dict[output_sensor.name] = np.append(
                    model_error_dict[output_sensor.name], me)

        return model_error_dict
