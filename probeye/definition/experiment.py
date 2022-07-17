# standard library
import copy as cp

# third party imports
import numpy as np


class Experiment:
    """
    An experiment.

    Parameters
    ----------
    name
        The name of the experiment, e.g. "Exp_20May.12".
    sensor_data
        The keys are the sensor's names (for example 'strain_gauge_1') and the
        values are the measured values.
    """

    def __init__(self, name: str, sensor_data: dict):

        # check types of input arguments
        if type(sensor_data) is not dict:
            raise TypeError(
                f"The sensor_data must be given as a dictionary. However, "
                f"found type '{type(sensor_data)}'."
            )
        for sensor_name, sensor_value in sensor_data.items():
            if type(sensor_value) not in [int, float, tuple, list, np.ndarray]:
                raise ValueError(
                    f"The sensor value of sensor '{sensor_name}' stated for experiment "
                    f"'{name}' is of type '{type(sensor_value)}'. However, only 'int', "
                    f"'float', 'tuple', 'list' or 'np.ndarray' are allowed."
                )

        # check that the sensor values are given in the right format; only floats, ints,
        # np.ndarray, lists and tuples are allowed; lists, np.ndarrays will be converted
        # to tuples; however, multidimensional np.ndarrays will lead to a value error
        sensor_data_tuples = cp.deepcopy(sensor_data)
        for sensor_name, values in sensor_data.items():
            if isinstance(values, list):
                if any(isinstance(element, list) for element in values):
                    raise ValueError(
                        f"The sensor values of an experiment must be given as 1D "
                        f"vectors. However, the sensor_data of '{sensor_name}' in "
                        f"experiment '{name}' is given as a nested list."
                    )
                sensor_data_tuples[sensor_name] = tuple(values)
            elif isinstance(values, np.ndarray):
                if values.ndim > 1:
                    raise ValueError(
                        f"The sensor values of an experiment must be given as 1D "
                        f"vectors. However, the sensor_data of '{sensor_name}' in "
                        f"experiment '{name}' are given as a {values.ndim}D array."
                    )
                if values.size == 1:
                    raise ValueError(
                        f"Encountered a np.ndarray with only one element in "
                        f"experiment '{name}', sensor '{sensor_name}'. Please use "
                        f"floats or integers to provide scalar data."
                    )
                sensor_data_tuples[sensor_name] = tuple(values)
            elif isinstance(values, (float, int)):
                sensor_data_tuples[sensor_name] = values

        # set the attributes of this class
        self.name = name
        self.sensor_data = sensor_data_tuples
        self.sensor_names = [*sensor_data_tuples.keys()]
        self.n_sensors = len(self.sensor_names)

        # the forward model will be assigned during InverseProblem.add_forward_model;
        # note that this attribute will only refer to the forward model's name
        self.forward_model = "not assigned"
