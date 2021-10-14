# standard library
import copy as cp
import numpy as np

# local imports
from probeye.subroutines import translate_prms_def
from probeye.subroutines import make_list
from probeye.subroutines import len_or_one


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

    def response(self, inp):
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

    def __call__(self, inp):
        return self.response(inp)

    def jacobian(self, inp, return_as_array=False):
        """
        Numerically computes the Jacobian matrix of the forward model and
        returns it in form of a dictionary or a numpy array. Note that this
        method should be overwritten, if there is a more efficient way to
        compute the jacobian.

        Parameters
        ----------
        inp : dict
            Contains both the exp. input data and the  model's parameters. The
            keys are the names, and the values are their numeric values.
        return_as_array : bool, optional
            When True, the Jacobian is returned as a numpy array. Otherwise,
            it is returned in form of a dictionary.

        Returns
        -------
        jac : dict or numpy.ndarray
            The Jacobian matrix in dict-form: The keys are the same as for the
            inp argument. The values are dictionaries with the model's output
            sensor names as keys and the derivatives as values. For example, the
            element jac['a']['y'] would give the derivative dy/da. If a numpy
            array is returned it can be interpreted as the conventional Jacobi
            matrix with respect to the columns and rows (i.e. the rows are the
            different gradients and the columns are derivatives with respect to
            one fixed parameter).
        """
        # eps is the machine precision; it is needed to compute the step size of
        # the central difference scheme below; note that this refers to single
        # precision (float32) since the processed arrays might be defined in
        # float32, in which case using the eps of double precision (float64)
        # would not work since the step size would be too small
        eps = np.finfo(np.float32).eps
        jac_dict = {}
        for prm_name, prm_value in inp.items():
            inp_left = cp.copy(inp)
            inp_right = cp.copy(inp)
            x = inp[prm_name]
            h = np.sqrt(eps) * x + np.sqrt(eps)
            inp_left[prm_name] = x - h
            inp_right[prm_name] = x + h
            dx = 2 * h
            response_dict_left = self(inp_left)
            response_dict_right = self(inp_right)
            jac_dict[prm_name] = {}
            for output_sensor in self.output_sensors:
                os_name = output_sensor.name
                jac_dict[prm_name][os_name] = (response_dict_right[os_name] -
                                               response_dict_left[os_name]) / dx
        # if requested, convert into array
        if return_as_array:
            n1 = len(self.output_sensors)
            n2 = len(inp)
            n3 = max([len_or_one(v) for v in [*inp.values()]])
            jac = np.zeros((n1, n2, n3))
            for j, osensor_dict in enumerate([*jac_dict.values()]):
                for i, derivative in enumerate([*osensor_dict.values()]):
                    jac[i, j, :] = derivative
        else:
            jac = jac_dict

        return jac
