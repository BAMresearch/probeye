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
    def __init__(self, name, x=0, y=0, z=0):
        """
        Parameters
        ----------
        name : str
            The name of the sensor, e.g. 'Deflection-Sensor bottom-left'.
        x : float, int, optional
            Positional x-coordinate of the sensor.
        y : float, int, optional
            Positional y-coordinate of the sensor.
        z : float, int, optional
            Positional z-coordinate of the sensor.
        """
        super().__init__(name)
        self.x = x
        self.y = y
        self.z = z
        self.xyz = np.array([x, y, z])
