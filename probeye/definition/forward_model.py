# standard library
from typing import Union, List, Tuple, Dict
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

    Parameters
    ----------
    name
        The name of the forward model. Must be unique among all forward model's names
        within a considered InverseProblem.
    """

    def __init__(
        self,
        name: str,
        _skip_interface_=False,
    ):

        # set the forward model's name
        self.name = name

        # this is just for consistency; values will be overwritten with the next command
        self.parameters = ["_self.parameters_not_set"]
        self.input_sensors = [Sensor("_self.input_sensors_not_set")]
        self.output_sensors = [Sensor("_self.output_sensors_not_set")]

        # set the three attributes above by running the user-defined method
        # self.interface; the exception triggered by naming the forward model '_dummy_'
        # is intended mostly for testing
        if name != "_dummy_":

            # now, run the user-defined 'interface'-method which will set the attributes
            # self.parameters, self.input_sensors and self.output_sensors
            self.interface()

            # check if self.parameters, self.input_sensors and self.output_sensors have
            # been set by the user in the required self.interface-method
            if self.parameters == ["_self.parameters_not_set"]:
                raise RuntimeError(
                    f"You did not set the required attribute 'self.parameters' in the "
                    f"forward model's 'interface'-method!"
                )
            if make_list(self.input_sensors)[0].name == "_self.input_sensors_not_set":
                raise RuntimeError(
                    "You did not set the required attribute 'self.input_sensors' in "
                    "the forward model's 'interface'-method!"
                )
            if make_list(self.output_sensors)[0].name == "_self.output_sensors_not_set":
                raise RuntimeError(
                    "You did not set the required attribute 'self.output_sensors' in "
                    "the forward model's 'interface'-method!"
                )
        self.prms_def, self.prms_dim = translate_prms_def(self.parameters)
        self.input_sensors = make_list(self.input_sensors)
        self.output_sensors = make_list(self.output_sensors)
        self.correlation_variables = self.check_sensor_correlation()

        # here, it is checked if the output sensors of the forward model share the same
        # error standard deviation parameters; this allows faster likelihood evaluations
        self.sensors_share_std_model = False
        self.sensors_share_std_measurement = False
        self.sensors_share_std_prms = False
        self.check_std_definitions()

        # set the attribute self.input_sensor for forward models with 1 input sensor
        self.input_sensor = Sensor("not_set_because_more_than_one_input_sensor")
        if len(self.input_sensors) == 1:
            self.input_sensor = self.input_sensors[0]

        # set the attribute self.output_sensor for forward models with 1 output sensor
        self.output_sensor = Sensor("not_set_because_more_than_one_output_sensor")
        if len(self.output_sensors) == 1:
            self.output_sensor = self.output_sensors[0]

        # ================================== #
        #   Attributes used/set by solvers   #
        # ================================== #

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

        # the following attribute is set by self.connect_experimental_data_to_sensors();
        # this method is called by the solver before solving the problem
        self.experiment_names = []  # type: list

        # the following attributes are set by the solver before solving the problem by
        # calling self.prepare_experimental_inputs_and_outputs()
        self.input_from_experiments = {}  # type: dict
        self.output_from_experiments = {}  # type: dict
        self.output_lengths = {}  # type: dict

    @property
    def input_sensor_names(self) -> List[str]:
        """Provides input_sensor_names attribute."""
        return [sensor.name for sensor in self.input_sensors]

    @property
    def input_sensor_dict(self) -> dict:
        """Returns dict with input sensor names as keys and sensor objects as values."""
        return {sensor.name: sensor for sensor in self.input_sensors}

    @property
    def input_channel_names(self) -> List[str]:
        """Provides input_channel_names attribute."""
        return self.input_sensor_names + [*self.prms_def.values()]

    @property
    def output_sensor_names(self) -> List[str]:
        """Provides input_sensor_names attribute."""
        return [sensor.name for sensor in self.output_sensors]

    @property
    def n_output_sensors(self) -> int:
        """Provides number of output_sensors as an attribute."""
        return len(self.output_sensor_names)

    @property
    def sensor_names(self) -> List[str]:
        """Provides a list of all sensor names as an attribute."""
        return self.input_sensor_names + self.output_sensor_names

    def check_sensor_correlation(self) -> list:
        """
        Checks if all output sensors share the same correlation variables, which is
        a requirement for a valid forward model definition. If this is the case, the
        common correlation variables are returned. Otherwise, an error is raised.

        Returns
        -------
        correlation_variables
            A list of strings (something like 't') or tuples (something like ('x', 'y'))
            stating the common correlation variables defined in the forward model's
            output sensors.
        """
        correlation_variables = self.output_sensors[0].correlation_variables
        for output_sensor in self.output_sensors:
            if output_sensor.correlation_variables != correlation_variables:
                raise RuntimeError(
                    f"The output sensors in forward model '{self.name}' do not share "
                    f"the same correlation variables!"
                )
        return correlation_variables

    def check_std_definitions(self):
        """
        Checks if the forward model's output sensors share a common model error and
        measurement error standard deviation parameter. The result is written to three
        of the forward model's attributes.
        """

        # first, check the model error standard deviation; the variable 'std_model_set'
        # will contain a set of all global parameter names for model error standard
        # deviations for the forward model's output sensors
        std_model_set = set()
        for output_sensor in self.output_sensors:
            std_model_set.add(output_sensor.std_model)
        if len(std_model_set) == 1:
            self.sensors_share_std_model = True

        # now, consider the measurement error standard deviation; the variable
        # 'std_measurement_set' will contain a set of all global parameter names for
        # measurement error standard deviations for the forward model's output sensors
        std_measurement_set = set()
        for output_sensor in self.output_sensors:
            std_measurement_set.add(output_sensor.std_measurement)
        if len(std_measurement_set) == 1:
            self.sensors_share_std_measurement = True

        # combine the information to a single flag
        self.sensors_share_std_prms = (
            self.sensors_share_std_model and self.sensors_share_std_measurement
        )

    def interface(self):
        """
        This method must be overwritten by the user. It is used to explicitly define the
        forward model's parameters, input and output sensors. Check out the integration
        tests to see examples.
        """
        raise NotImplementedError(
            f"No 'interface'-method defined for forward model '{self.name}'!"
        )

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

    def connect_experimental_data_to_sensors(self, exp_name: str, sensor_values: dict):
        """
        Connects the experimental data from an experiments to the corresponding sensors
        of the forward model. Note that sensor-objects are essentially dictionaries, so
        the connection is established by adding the 'exp_name' as key to the respective
        sensor-(dict)-object with the measurements as the dict-values. There are no
        checks in this method because it is only used by InverseProblem.add_experiment
        which already does the consistency checks before calling this method.

        Parameters
        ----------
        exp_name
            The name of the experiment the 'sensor_values' are coming from.
        sensor_values
            Keys are the sensor names (like "x" or "y") and values are either floats,
            integers or numpy-ndarrays representing the measured values.
        """

        # connect the forward model's input sensors to the experiments
        for sensor in self.input_sensors:
            sensor[exp_name] = sensor_values[sensor.name]

        # connect the forward model's output sensors to the experiments
        for sensor in self.output_sensors:
            sensor[exp_name] = sensor_values[sensor.name]

        # collect all connected experiments to a separate list for convenience
        self.experiment_names.append(exp_name)

    def prepare_experimental_inputs_and_outputs(self):
        """
        This method prepares the experimental-data-collection over the forward model's
        input and output sensors. This is done in an own method here for efficiency
        reasons. Without this method, the loops over the input and output sensors would
        be repeated in each evaluation of the forward model. This method is called in
        the solvers before starting an inference routine. It sets the two general
        attributes 'self.input_from_experiments' and 'self.output_from_experiment' as
        well as the correlation-related attribute 'self.output_lengths'.
        """

        # set 'self.input_from_experiments' and 'self.output_from_experiments'; both
        # attributes are dictionaries with the same structure; a simple example could
        # look like: {'Exp1': {'deflection_1': np.array([1.9, 2.3]), 'F': 1200.0}}
        for exp_name in self.experiment_names:
            exp_inp = {}
            for input_sensor in self.input_sensors:
                exp_inp[input_sensor.name] = input_sensor[exp_name]
            self.input_from_experiments[exp_name] = exp_inp
            exp_out = {}
            for output_sensor in self.output_sensors:
                exp_out[output_sensor.name] = output_sensor[exp_name]
            self.output_from_experiments[exp_name] = exp_out

        # set the self.output_lengths dictionary; this dict is required for the methods
        # self.std_model and self.std_measurement; it contains information on the length
        # of the returned values of the forward model in the different experiments; a
        # simple example for an uncorrelated case could look like this:
        # {'Ex1': {'': {'total': 202, 'increments': [101, 101], 'names': ['y1', 'y2']}},
        #  'Ex2': {'': {'total': 102, 'increments': [51, 51], 'names': ['y1', 'y2']}}}
        # this is interpreted as follows: for experiment 1 (named 'Ex1') the forward
        # model's output dictionary will eventually be translated into a vector holding
        # 202 values, where the first 101 belong to output sensor 'y1' and the following
        # 101 values belong to output sensor 'y2'; an analogous interpretation holds for
        # the second experiment (named 'Ex2'); in a correlated case, the created dict
        # will additionally contain the lengths of the correlation variables, e.g.:
        # {'Ex1': {'':  {'total': 12, 'increments': [6, 6], 'names': ['y1', 'y2']},
        #          't': {'total': 2,  'increments': [1, 1], 'names': ['y1', 'y2']},
        #          'x': {'total': 12, 'increments': [6, 6], 'names': ['y1', 'y2']}}
        # the 't' and 'x' entries are interpreted as the 't'-correlation vector having
        # length 2 and the 'x'-correlation vector having length 12, while the remaining
        # information is interpreted analogously as described before
        output_lengths = {}  # type: dict
        for exp_name in self.experiment_names:
            output_lengths[exp_name] = {}  # type: dict
            # add the information for the model response
            output_lengths[exp_name][""] = {
                "total": 0,
                "increments": [],
                "names": [],
            }
            for output_sensor in self.output_sensors:
                n_i = len_or_one(output_sensor[exp_name])
                name = output_sensor.name
                output_lengths[exp_name][""]["increments"].append(n_i)
                output_lengths[exp_name][""]["names"].append(name)
            output_lengths[exp_name][""]["total"] = sum(
                output_lengths[exp_name][""]["increments"]
            )
            # add the information for the correlation vectors
            for corr_var_ in self.correlation_variables:
                corr_var_tuple = corr_var_
                if isinstance(corr_var_, str):
                    corr_var_tuple = (corr_var_,)
                for corr_var in corr_var_tuple:
                    output_lengths[exp_name][corr_var] = {
                        "total": 0,
                        "increments": [],
                        "names": [],
                    }
                    for output_sensor in self.output_sensors:
                        n_i = output_sensor.corr_var_lengths[exp_name][corr_var]
                        name = output_sensor.name
                        output_lengths[exp_name][corr_var]["increments"].append(n_i)
                        output_lengths[exp_name][corr_var]["names"].append(name)
                    output_lengths[exp_name][corr_var]["total"] = sum(
                        output_lengths[exp_name][corr_var]["increments"]
                    )
        self.output_lengths = output_lengths

    def std_values(
        self,
        prms: dict,
        exp_name: str,
        corr_var: str = "",
        measurement_error=False,
    ) -> Tuple[
        Union[int, float, np.ndarray], Union[int, float, np.ndarray, None], bool
    ]:
        """
        Returns the model/measurement error standard deviations either as scalar (if all
        sensors share the same model/measurement error standard deviation) or as vectors
        expanded to the length of the requested correlation variable vector.

        Parameters
        ----------
        prms
            The input parameter dictionary.
        exp_name
            The name of the considered experiment.
        corr_var
            The correlation variable the vector should be expanded to. If no correlation
            is defined, this variable must be an empty string. In this case, a returned
            vector would be expanded to the full vectorized forward model's response.
        measurement_error
            True, if also a measurement error should be considered. Otherwise, False.

        Returns
        -------
        std_model
            Either a scalar, or a vector with the forward model's model error standard
            deviation expanded to the required length defined by the  given experiment.
        std_measurement
            Either a scalar, or a vector with the forward model's measurement error std.
            deviation expanded to the required length defined by the given experiment.
        stds_are_scalar
            True, if the returned standard deviations are scalars, otherwise False.
        """
        std_measurement = None
        if self.sensors_share_std_prms:
            # in this case, all sensors have the same parameter that describes their
            # model/measurement error; hence, we just need a scalar value
            std_model = prms[self.output_sensors[0].std_model]
            if measurement_error:
                std_measurement = prms[self.output_sensors[0].std_measurement]
            stds_are_scalar = True
        else:
            # in this case, the forward model has more than one sensor, and not all of
            # those sensors share the same 'std_model' and/or 'std_measurement'
            # attribute; so, we have to assemble non-constant vector(s) with different
            # values for std_model or std_measurement
            idx_start = 0
            std_model = np.zeros(self.output_lengths[exp_name][corr_var]["total"])
            if measurement_error:
                std_measurement = np.zeros(
                    self.output_lengths[exp_name][corr_var]["total"]
                )
            increments = self.output_lengths[exp_name][corr_var]["increments"]
            for output_sensor, n_i in zip(self.output_sensors, increments):
                idx_end = idx_start + n_i
                std_model[idx_start:idx_end] = prms[output_sensor.std_model]
                if measurement_error:
                    std_meas = prms[output_sensor.std_measurement]
                    std_measurement[idx_start:idx_end] = std_meas  # type: ignore
                idx_start = idx_end
            stds_are_scalar = False
        return std_model, std_measurement, stds_are_scalar
