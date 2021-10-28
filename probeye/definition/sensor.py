# third party imports
from loguru import logger

# local imports
from probeye.subroutines import process_spatial_coordinates


class Sensor:
    """
    Base class for a sensor of the forward model. In its simplest form just a
    class with a 'name' attribute. Additional attributes for the measured
    quality (measurand) and the corresponding unit can be defined. If the
    sensors position(s) are important, they can be defined as attributes.
    Further attributes can be defined by the user by creating new classes
    derived from this one.
    """

    @logger.catch(reraise=True)
    def __init__(self, name, measurand='not defined', unit='not defined',
                 x=None, y=None, z=None, coords=None, order=('x', 'y', 'z')):
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

        # basic attributes
        self.name = name
        self.measurand = measurand
        self.unit = unit

        # translate the input to a coords-array
        self.coords, self._order = process_spatial_coordinates(
            x=x, y=y, z=z, coords=coords, order=order)

        # this contains the information which row contains which coordinate
        self.index_dict = {coord: i for i, coord in enumerate(self._order)}

    @property
    def x(self):
        """Provides x-coords as attribute without copying them from coords."""
        return self.coords[self.index_dict['x']] if 'x' in self._order else None

    @property
    def y(self):
        """Provides y-coords as attribute without copying them from coords."""
        return self.coords[self.index_dict['y']] if 'y' in self._order else None

    @property
    def z(self):
        """Provides z-coords as attribute without copying them from coords."""
        return self.coords[self.index_dict['z']] if 'z' in self._order else None

    @property
    def order(self):
        """Provides read-access to privat attribute self._order."""
        return self._order
