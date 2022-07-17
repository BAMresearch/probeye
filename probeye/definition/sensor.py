# standard library
from typing import Union, List, Tuple, Optional

# third party imports
import numpy as np

# local imports
from probeye.subroutines import process_spatiotemporal_coordinates
from probeye.subroutines import len_or_one


class Sensor(dict):
    """
    Base class for an input or output sensor of the forward model. In its simplest form
    an instance of this class is just a dictionary with a 'name' attribute. Additional
    attributes for the measured quality (measurand) and the corresponding unit can be
    defined as well. If the sensors position(s) are important, they can be defined as
    attributes. Further attributes can be defined by the user by creating new classes
    derived from this one. If the sensor is used as an output sensor, the parameters
    that describe the statistics of the error in this sensor must be given. Moreover,
    a sensor objects points to the experimental data it refers to. For that purpose,
    the sensor class is derived from the dictionary class (so, essentially, a sensor is
    a dictionary with additional attributes). The keys of an output sensor are the
    experiment's names in which some data for this sensor was collected. Consequently,
    the values are the measured values of the sensor in the respective experiment.

    Parameters
    ----------
    name
        The name of the sensor, for example 'T_in', 'v_x' or just 'y'. Note that the
        sensor's name must be unique among all the sensors of the forward model the
        sensor was defined on. The name of the sensor does not have to be unique across
        multiple forward models though. So, for example, if there are two forward models
        defined in an inverse problem, they can both have a sensor named 'y'.
    measurand
        Defines what the sensor measures, for example 'temperature' or 'deflection'.
        Note that this is optional information, which is currently not used elsewhere
        in the code.
    unit
        Defines what unit is associated with the sensor's measurements, for example
        'mm' for a deflection or 'K' for a temperature.
    std_model
        The name of the globally defined parameter that describes the standard deviation
        of the model prediction error in this sensor. If multiple output sensors are
        defined on a forward model, these sensors can have different parameters that
        describe their model prediction error..
    x
        x-coordinate(s) of the sensor. When given, the coords- argument must be None.
        Usually, this value will be a scalar (for example if the sensor represents a
        point-like sensor like a strain gauge), but the value can also be a vector. This
        can make sense, for example, for an optical measurement system, which tracks the
        deflections of multiple points on a loaded structure. If next to 'x' also 'y'
        and/or 'z' are given, they must all have the same length.
    y
        y-coordinate(s) of the sensor. When given, the coords- argument must be None.
        Check out the explanations of 'x' for more information.
    z
        z-coordinate(s) of the sensor. When given, the coords-argument must be None.
        Check out the explanations of 'x' for more information.
    coords
        Some or all of the coordinates x, y, z concatenated as an array. Each row
        corresponds to a constant coordinate, for example the first row might contain
        all values for the x-coordinate of all points. Which row corresponds to which
        coordinate is defined via the order-argument. When 'coords' given, the arguments
        'x', 'y' and 'z' must be None.
    order
        Only relevant when 'coords' is given. Defines which row in 'coords' corresponds
        to which coordinate. For example, order=('x', 'y', 'z') means that the 1st row
        of 'coords' contains x-coordinates while the 2nd row contains y-coords and the
        3rd row contains the z-coordinates.
    """

    def __init__(
        self,
        name: str,
        measurand: str = "not defined",
        unit: str = "not defined",
        x: Union[float, int, np.ndarray, None] = None,
        y: Union[float, int, np.ndarray, None] = None,
        z: Union[float, int, np.ndarray, None] = None,
        coords: Optional[np.ndarray] = None,
        order: Tuple[str, ...] = ("x", "y", "z"),
        std_model: str = "not defined",
    ):

        # basic attributes
        self.name = name
        self.measurand = measurand
        self.unit = unit

        # translate possibly given coordinate-information to a coords-array
        self.coords, self._order = process_spatiotemporal_coordinates(
            x=x, y=y, z=z, coords=coords, order=order
        )

        # this contains the information which row contains which coordinate
        self.index_dict = {coord: i for i, coord in enumerate(self._order)}

        # these two attributes contain the global names of parameters that describe the
        # model error and the measurement error in this sensor respectively; they will
        # be referred to when evaluating the likelihood function of a likelihood model
        self.std_model = std_model
        self.std_measurement = "not defined"

        # required due to inheriting from dict-class
        super().__init__()

    @property
    def x(self) -> Union[np.ndarray, None]:
        """Provides x-coords as attribute without copying them from coords."""
        return self.coords[self.index_dict["x"]] if "x" in self._order else None

    @property
    def y(self) -> Union[np.ndarray, None]:
        """Provides y-coords as attribute without copying them from coords."""
        return self.coords[self.index_dict["y"]] if "y" in self._order else None

    @property
    def z(self) -> Union[np.ndarray, None]:
        """Provides z-coords as attribute without copying them from coords."""
        return self.coords[self.index_dict["z"]] if "z" in self._order else None

    @property
    def order(self) -> List[str]:
        """Provides read-access to privat attribute self._order."""
        return self._order

    def __setitem__(self, key: str, value: Union[int, float, np.ndarray]):
        """
        Adds a key-value pair to the sensor instance (remember, a sensor is essentially
        a dictionary with additional attributes). The key is the name of an experiment
        while the value is a numeric measurement (vector) recorded by this sensor.

        Parameters
        ----------
        key
            The key of the key-value pair to be added to self. In this case, the key is
            the name of an experiment.
        value
            The value of the key-value pair to be added to self. In this case, the value
            is the measurement (vector) recorded by the sensor in the given experiment.
        """

        # finally, add the key-value-pair to self
        super().__setitem__(key, value)

    def __str__(self) -> str:
        """
        Allows to print the sensor via print(sensor) if sensor is an instance of Sensor.
        """
        return f"Sensor(name='{self.name}')"
