# standard library
from typing import List

# third party imports
from loguru import logger

# local imports
from probeye.definition.sensor import Sensor
from probeye.subroutines import translate_prms_def
from probeye.subroutines import make_list


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
    ):

        # set the forward model's name
        self.name = name

        # this is just for consistency; values will be overwritten with the next command
        self.parameters = ["_self.parameters_not_set"]
        self.input_sensors = [Sensor("_self.input_sensors_not_set")]
        self.output_sensors = [Sensor("_self.output_sensors_not_set")]
        self.prms_def = {}  # type: dict
        self.prms_dim = 0

        # overwrite the attr. above by running the user-defined method self.interface
        self._evaluate_interface()

        # here, it is checked if the output sensors of the forward model share the same
        # model error std. dev. parameters; this allows faster likelihood evaluations
        self.sensors_share_std_model = False
        self._check_std_definitions()

        # ================================== #
        #   Attributes used/set by solvers   #
        # ================================== #

        # the following attribute is set by self.connect_experimental_data_to_sensors();
        # this method is called by the solver before solving the problem
        self.experiment_names = []  # type: list

        # the following attributes are set by the solver before solving the problem by
        # calling self.prepare_experimental_inputs_and_outputs()
        self.input_from_experiments = {}  # type: dict
        self.output_from_experiments = {}  # type: dict
        self.output_lengths = {}  # type: dict

    @property
    def input_sensor(self) -> Sensor:
        """Returns the 1st input sensor. Intended for models with only one onf them."""
        if len(self.input_sensors) > 1:
            logger.warning(
                f"You used the property 'input_sensor' which is intended for forward "
                f"models with only one input sensor. However, the forward model "
                f"'{self.name}' has {len(self.input_sensors)} input sensors."
            )
        return self.input_sensors[0]

    @property
    def output_sensor(self) -> Sensor:
        """Returns the 1st output sensor. Intended for models with only one onf them."""
        if len(self.output_sensors) > 1:
            logger.warning(
                f"You used the property 'output_sensor' which is intended for forward "
                f"models with only one output sensor. However, the forward model "
                f"'{self.name}' has {len(self.output_sensors)} output sensors."
            )
        return self.input_sensors[0]

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

    def _evaluate_interface(self):
        """
        Sets the attributes prms_def, prms_dim, input_sensors and output_sensors. This
        method is called during initialization.
        """

        # the exception triggered by naming the forward model '_dummy_' is intended
        # mostly for testing
        if self.name != "_dummy_":

            # now, run the user-defined 'ontology'-method which will set the attributes
            # self.parameters, self.input_sensors and self.output_sensors
            self.interface()

            # check if self.parameters, self.input_sensors and self.output_sensors have
            # been set by the user in the required self.ontology-method
            if self.parameters == ["_self.parameters_not_set"]:
                raise RuntimeError(
                    f"You did not set the required attribute 'self.parameters' in the "
                    f"forward model's 'interface'-method!"
                )
            if len(self.input_sensors) > 0:
                inp_sensors = self.input_sensors  # just to avoid line-break
                if make_list(inp_sensors)[0].name == "_self.input_sensors_not_set":
                    raise RuntimeError(
                        "You did not set the required attribute 'self.input_sensors' "
                        "in the forward model's 'interface'-method!"
                    )
            if make_list(self.output_sensors)[0].name == "_self.output_sensors_not_set":
                raise RuntimeError(
                    "You did not set the required attribute 'self.output_sensors' in "
                    "the forward model's 'interface'-method!"
                )
        self.prms_def, self.prms_dim = translate_prms_def(self.parameters)
        self.input_sensors = make_list(self.input_sensors)
        self.output_sensors = make_list(self.output_sensors)

    def _check_std_definitions(self):
        """
        Checks if the forward model's output sensors share a common model error standard
        deviation parameter. The result is written to self.sensors_share_std_model.
        """
        # first, check the model error standard deviation; the variable 'std_model_set'
        # will contain a set of all global parameter names for model error standard
        # deviations for the forward model's output sensors
        std_model_set = set()
        for output_sensor in self.output_sensors:
            std_model_set.add(output_sensor.std_model)
        if len(std_model_set) == 1:
            self.sensors_share_std_model = True

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

    def connect_experimental_data_to_sensors(self, exp_name: str, sensor_data: dict):
        """
        Connects the experimental data from an experiments to the corresponding sensors
        of the forward model. Note that sensor-objects are essentially dictionaries, so
        the connection is established by adding the 'exp_name' as key to the respective
        sensor-(dict)-object with the measurements as the dict-values. This method is
        called in the solvers before starting an inference routine.

        Parameters
        ----------
        exp_name
            The name of the experiment the 'sensor_values' are coming from.
        sensor_data
            Keys are the sensor names (like "x" or "y") and values are either floats,
            integers or numpy-ndarrays representing the measured values.
        """

        # connect the forward model's input sensors to the experiments
        for sensor in self.input_sensors:
            sensor[exp_name] = sensor_data[sensor.name]

        # connect the forward model's output sensors to the experiments
        for sensor in self.output_sensors:
            sensor[exp_name] = sensor_data[sensor.name]

        # collect all connected experiments to a separate list for convenience
        self.experiment_names.append(exp_name)

    def prepare_experimental_inputs_and_outputs(self):
        """
        This method prepares the experimental-data-collection over the forward model's
        input and output sensors. This is done in an own method here for efficiency
        reasons. Without this method, the loops over the input and output sensors would
        be repeated in each evaluation of the forward model. This method is called in
        the solvers before starting an inference routine. It sets the two general
        attributes 'self.input_from_experiments' and 'self.output_from_experiment'.
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
