# third party imports
import numpy as np


class Sensor:
    """
    Base class for a sensor of the forward model. Essentially just a class with
    a 'name' attribute. Additional attributes (such as sensor position, its
    local temperature, etc.) can be added by the user. See e.g. the definition
    of PositionSensor given below.
    """
    def __init__(self, name):
        """
        Parameters
        ----------
        name : str
            The name of the sensor, e.g. 'Deflection-Sensor bottom-left'.
        """
        self.name = name


class PositionSensor(Sensor):
    """
    Class for a forward model's sensor with constant positional coordinates.
    """
    def __init__(self, name, x=None, y=None, z=None):
        """
        Parameters
        ----------
        name : str
            The name of the sensor, e.g. 'Deflection-Sensor bottom-left'.
        x : float, int, None, optional
            Positional x-coordinate of the sensor.
        y : float, int, None, optional
            Positional y-coordinate of the sensor.
        z : float, int, None, optional
            Positional z-coordinate of the sensor.
        """
        super().__init__(name)
        # check that at least one coordinate is given
        if (x is None) and (y is None) and (z is None):
            raise RuntimeError(
                "At least one coordinate of x, y and z have to be specified. "
                "You did not specify any of those.")

        # write the single coordinates to attributes when given
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if z is not None:
            self.z = z

        # provide a vector of the specified coordinates
        self.coords = np.array([v for v in [x, y, z] if v is not None])
