# standard library
from typing import Union, List, Dict
import copy as cp
import numpy as np

# local imports
from probeye.definition.sensor import Sensor
from probeye.subroutines import translate_prms_def
from probeye.subroutines import make_list
from probeye.subroutines import len_or_one


class ForwardModelBase:
    """
    This class serves as a base class for any forward model. When you want to define a
    specific forward model, you need to derive your own class from this one, and then
    define the '__call__' method. The latter essentially describes the model function
    mapping the model input to the output.
    """

    def __init__(
        self,
        prms_def_: Union[str, List[Union[str, dict]], dict],
        input_sensors: Union[Sensor, List[Sensor]],
        output_sensors: Union[Sensor, List[Sensor]],
    ):
        """
        Parameters
        ----------
        prms_def_
            Contains the model's latent parameter names. The list may only contain
            strings or one-element dictionaries. It could look, for example, like
            [{'a': 'm'}, 'b']. The one-element dictionaries account for the possibility
            to define a local name for a latent parameter that is different from the
            global name. In the example above, the latent parameter with the global name
            'a' will be referred to as 'm' within the model. So, the one-element dicts
            have the meaning {<global name>: <local name>}. String-elements are
            interpreted as having similar local and global names. Note that the local-
            name option will not be required most of the times. The input from global to
            local name can also be provided as a dict. In the example above it would
            look like {'a': 'm', 'b': 'b'}.
        input_sensors
            Contains sensor-objects structuring the model input.
        output_sensors
            Contains sensor-objects structuring the model output.
        """

        # convert the given parameter names to a dictionary with global names as keys
        # and local names as values
        self.prms_def, self.prms_dim = translate_prms_def(prms_def_)

        # other attributes
        self.input_sensors = make_list(input_sensors)
        self.output_sensors = make_list(output_sensors)

        # this attribute might be used to write the forward model's input structure to;
        # it has the same structure like the 'inp' argument of the response method, but
        # instead of the input channel's values it states the input channels number of
        # elements
        self.input_structure = {ic: 0 for ic in self.input_channel_names}

        # this attributes might be used by inference engines that need a forward model
        # wrapper, which only returns numeric vectors; for reconstructing the response
        # dictionary from the numeric vector, one needs to know the response
        # dictionary's structure; this dictionaries will then contain the same keys as
        # the response method's return dictionary, while the values will be the number
        # of elements contained in the values; e.g., {'x': np.array([0, 0.1, 0.2]),
        # 'a': 3.7} will have a structure  of {'x': 3, 'a': 1}; this attr. is not used
        # by all inference engines
        self.response_structure = {os.name: 0 for os in self.output_sensors}

    @property
    def input_sensor_names(self) -> List[str]:
        """Provides input_sensor_names attribute."""
        return [sensor.name for sensor in self.input_sensors]

    @property
    def input_channel_names(self) -> List[str]:
        """Provides input_channel_names attribute."""
        return self.input_sensor_names + [*self.prms_def.values()]

    @property
    def output_sensor_names(self) -> List[str]:
        """Provides input_sensor_names attribute."""
        return [sensor.name for sensor in self.output_sensors]

    @property
    def sensor_names(self) -> List[str]:
        """Provides a list of all sensor names as an attribute."""
        return self.input_sensor_names + self.output_sensor_names

    def response(self, inp: dict) -> dict:
        """
        Evaluates the model response and provides computed results for all of the
        model's output sensors. This method must be overwritten by the user.

        Parameters
        ----------
        inp
            Contains both the exp. input data and the  model's parameters. The keys are
            the names, and the values are their numeric values.

        Returns
        -------
        dict
            Contains the model response (value) for each output sensor, referenced by
            the output sensor's name (key).
        """
        raise NotImplementedError(
            "Your model does not have a proper 'response'-method yet. You need  to "
            "define this method, so you can evaluate your model."
        )

    def __call__(self, inp: dict) -> dict:
        """
        Calls the self.response method. Shortens internal forward model calls.
        """
        return self.response(inp)

    def jacobian(self, inp: dict) -> Dict[str, dict]:
        """
        Numerically computes the Jacobian matrix of the forward model and returns it in
        form of a dictionary. Note that this method should be overwritten, if there is a
        more efficient way to compute the jacobian, for example, when one can compute
        the Jacobian analytically.

        Parameters
        ----------
        inp
            Contains both the exp. input data and the  model's parameters. The keys are
            the names, and the values are their numeric values.

        Returns
        -------
        jac_dict
            The Jacobian matrix in dict-form: The keys are the names of the forward
            model's output sensors. The values are dictionaries with the forward model's
            input channel's names as keys and the derivatives or Nones as values.
            Derivatives are only provided for the model's parameters, see self.prms_def.
            For all other input channels (e.g., measurements from an experiment) None is
            written to the dictionary's values, since they are not required by sampling
            routines. To give an example: the element jac['y']['a'] would give the
            derivative dy/da, and jac['y'] would give the gradient of the fwd. model's
            y-computation with respect to the input channels in a dictionary-format.
        """
        # eps is the machine precision; it is needed to compute the step size of the
        # central difference scheme below; note that this refers to single precision
        # (float32) since the processed arrays might be defined in float32, in which
        # case using the eps of double precision (float64) would not work since the
        # step size would be too small
        eps = np.finfo(np.float32).eps
        # the following evaluations are needed in the for-loop; they are put here so
        # they are not repeatedly evaluated (to the same value) during the for-loop
        sqrt_eps = np.sqrt(eps)
        response_dict_center = self.response(inp)
        inp_right = cp.deepcopy(inp)  # required to prevent side effects
        # prepare the dictionary; this structure needs to be external from the main loop
        # below since the filling of the dictionary could only be efficiently done in
        # the format jac_dict[prm_name][os_name] which is less readable; the format
        # created in the implemented way is easier to to read since jac['y']['a']
        # corresponds to dy/da in jac['y'] is the gradient of y with respect to theta
        jac_dict = {}  # type: Dict[str, dict]
        for output_sensor in self.output_sensors:
            jac_dict[output_sensor.name] = {}
            for prm_name in inp.keys():
                if prm_name in self.prms_def.values():  # values are local names
                    nrows = len_or_one(response_dict_center[output_sensor.name])
                    ncols = len_or_one(inp[prm_name])
                    value = np.zeros((nrows, ncols))  # type: Union[np.ndarray, None]
                else:
                    value = None
                jac_dict[output_sensor.name][prm_name] = value
        for prm_name, prm_value in inp.items():
            # derivatives only need to be computed for the model's parameters, not for
            # the input that comes from the experiments; e.g., in case of y=mx+b with
            # parameters m, b, one does not need dy/dx
            if prm_name not in self.prms_def.values():  # values are local names
                continue
            # the following loop accounts for the fact, that parameters can be
            # multidimensional
            for i, x in enumerate(np.atleast_1d(inp[prm_name])):
                # the following formula for the step size is NOT taken from the
                # literature; in the literature, a common recommended choice for the
                # step size h given x is not 0 is h = sqrt_eps * x, see for example:
                # https://en.wikipedia.org/wiki/Numerical_differentiation; we added the
                # term '+ sqrt_eps' below to also cover the cases where x actually is
                # zero (or very close to 0)
                h = sqrt_eps * x + sqrt_eps
                inp_right[prm_name] = x + h
                response_dict_right = self.response(inp_right)
                for output_sensor in self.output_sensors:
                    # the simple forward scheme should be sufficient for most
                    # applications since the Jacobian will only be used as info for
                    # choosing the next sample; for that purpose it is secondary if it
                    # contains small numerical errors
                    jac_dict[output_sensor.name][prm_name][:, i] = (
                        response_dict_right[output_sensor.name]
                        - response_dict_center[output_sensor.name]
                    ) / h
                inp_right[prm_name] = inp[prm_name]  # resetting perturbed value
        return jac_dict

    def jacobian_dict_to_array(
        self,
        inp: dict,
        jac_dict: dict,
        n_inp_dim: int,
    ) -> np.ndarray:
        """
        Converts the Jacobian in dict-format (computed by the above 'jacobian' method)
        into a numpy array. This method is external to the above 'jacobian' method, so
        that it is easier for a user to overwrite the it (i.e., the 'jacobian' method)
        without also having to define the conversion into an array.

        Parameters
        ----------
        inp
            See docstring of the 'jacobian'-method above.
        jac_dict
            See docstring of the 'jacobian'-method above.
        n_inp_dim
            The added-up dimensions of the forward model's input channels, i.e., of all
            model parameters and other input variables.

        Returns
        -------
        jac
            Similar structure as the conventional Jacobi matrix with respect to the
            columns and rows (i.e. the rows are the different gradients and the columns
            are derivatives with respect to one fixed parameter).
        """

        # n1 is the number of the forward model's output sensors; n2 is the dimension of
        # the forward model's input channels, i.e., the added up dimensions of the input
        # sensors and the number of the forward model's parameters; finally, n3 is the
        # maximum number of elements in the n2 input channels; the model's parameters
        # are usually scalars, but the input sensors might be vectors with more than one
        # element
        n1 = len(self.output_sensors)
        n2 = n_inp_dim
        n3 = max([len_or_one(v) for v in inp.values()])
        jac = np.zeros((n1 * n3, n2))
        for i, prm_dict in enumerate(jac_dict.values()):
            idx_start = i * n3
            j = 0
            for derivative in prm_dict.values():
                if derivative is None:
                    # in this case, the input variable is not a parameter; the
                    # corresponding column in the Jacobian will remain zero, which is
                    # ok, since it won't be used to compute anything
                    j += 1
                    continue
                nvals = derivative.shape[0]
                ncomp = derivative.shape[1]
                idx_end = idx_start + nvals
                jac[idx_start:idx_end, j : (j + ncomp)] = derivative
                j += ncomp
        return jac
