# third party imports
import numpy as np

# local imports
from probeye.subroutines import make_list


class Sensor:
    """
    Base class for a sensor of the forward model. Essentially just a class with
    a 'name' attribute. Additional attributes (such as sensor position, its
    local temperature, etc.) can be added by the user. See e.g. the definition
    of PositionSensor given below.
    """
    def __init__(self, name, measurand="not defined", unit="not defined"):
        """
        Parameters
        ----------
        name : str
            The name of the sensor, e.g. 'Deflection-Sensor bottom-left'.
        measurand : str, optional
            Defines what the sensor measures, e.g. 'deflection'.
        unit : str, optional
            Defines what unit is associated with the sensor's measurements, for
            example 'mm'.
        """
        self.name = name
        self.measurand = measurand
        self.unit = unit


class PositionSensor(Sensor):
    """
    Class for a forward model's sensor with constant positional coordinates.
    """
    def __init__(self, name, measurand='position', unit='', x=None, y=None,
                 z=None, coords=None, order=('x', 'y', 'z')):
        """
        Parameters
        ----------
        name : str
            The name of the sensor, e.g. 'Deflection-Sensor bottom-left'.
        measurand : str, optional
            Defines what the sensor measures.
        unit : str, optional
            Defines what unit is associated with the sensor's measurements, for
            example 'mm'.
        x : float, int, numpy.ndarray, None, optional
            Positional x-coordinate of the sensor. When given, the coords-
            argument must be None.
        y : float, int, numpy.ndarray, None, optional
            Positional y-coordinate of the sensor. When given, the coords-
            argument must be None.
        z : float, int, numpy.ndarray, None, optional
            Positional z-coordinate of the sensor. When given, the coords-
            argument must be None.
        coords : numpy.ndarray, optional
            Some or all of the coordinates x, y, z concatenated as an array.
            Each row corresponds to a constant coordinate. Which row corresponds
            to which coordinate is defined via the order-argument. When given,
            the arguments x, y and z must be None.
        order : tuple[str], optional
            Only relevant when coords is given. Defines which row in coords
            corresponds to which coordinate. For example, order=('x', 'y', 'z')
            means that the 1st row are x-coordinates, the 2nd row are y-coords
            and the 3rd row are the z-coordinates.
        """
        super().__init__(name, measurand=measurand, unit=unit)

        # check that at least one coordinate is given
        if (x is None) and (y is None) and (z is None) and (coords is None):
            raise RuntimeError(
                "At least one coordinate of x, y and z or a coordinate array"
                "(coords) has to be specified. You did not specify anything.")

        # define the attributes 'order' and 'coords'
        if coords is None:
            # in this case, the coordinates are given directly via x, y, z; note
            # that due to the eval-statement, the following for-loop cannot be
            # put into a list comprehension
            self.order = []  # this is going to be a list of str like ['x', 'z']
            self.coords = []
            for v in order:
                if eval(v) is not None:
                    self.order.append(v)
                    self.coords.append(make_list(eval(v)))
            self.coords = np.array(self.coords)
        else:
            if not ((x is None) and (y is None) and (z is None)):
                raise RuntimeError(
                    "When 'coords' is provided as an argument, you cannot "
                    "provide 'x', 'y' and 'z' at the same time!")

            # here, the coords-array is given directly; the order is taken from
            # the order-argument
            self.order = order[:coords.shape[0]]
            self.coords = coords

        # this contains the information which row contains which coordinate
        self.index_dict = {coord: i for i, coord in enumerate(self.order)}

    @property
    def x(self):
        """Provides x-coords as attribute without copying them from coords."""
        return self.coords[self.index_dict['x']] if 'x' in self.order else None

    @property
    def y(self):
        """Provides y-coords as attribute without copying them from coords."""
        return self.coords[self.index_dict['y']] if 'y' in self.order else None

    @property
    def z(self):
        """Provides z-coords as attribute without copying them from coords."""
        return self.coords[self.index_dict['z']] if 'z' in self.order else None
