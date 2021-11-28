# standard library
from typing import Union, List, Tuple, Optional

# third party imports
import numpy as np

# local imports
from probeye.subroutines import process_spatiotemporal_coordinates


class Sensor:
    """
    Base class for a sensor of the forward model. In its simplest form just a class with
    a 'name' attribute. Additional attributes for the measured quality (measurand) and
    the corresponding unit can be defined. If the sensors position(s) are important,
    they can be defined as attributes. Further attributes can be defined by the user by
    creating new classes derived from this one.
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
    ):
        """
        Parameters
        ----------
        name
            The name of the sensor, e.g. 'Deflection-Sensor bottom-left'.
        measurand
            Defines what the sensor measures, e.g. 'deflection'.
        unit
            Defines what unit is associated with the sensor's measurements, e.g. 'mm'.
        x
            x-coordinate of the sensor. When given, the coords- argument must be None.
        y
            y-coordinate of the sensor. When given, the coords- argument must be None.
        z
            z-coordinate of the sensor. When given, the coords-argument must be None.
        coords
            Some or all of the coordinates x, y, z concatenated as an array. Each row
            corresponds to a constant coordinate. Which row corresponds to which
            coordinate is defined via the order-argument. When given, the arguments x, y
            and z must be None.
        order
            Only relevant when coords is given. Defines which row in coords corresponds
            to which coordinate. For example, order=('x', 'y', 'z') means that the 1st
            row are x-coordinates, the 2nd row are y-coords and the 3rd row are the
            z-coordinates.
        """

        # basic attributes
        self.name = name
        self.measurand = measurand
        self.unit = unit

        # translate the input to a coords-array
        self.coords, self._order = process_spatial_coordinates(
            x=x, y=y, z=z, coords=coords, order=order
        )

        # this contains the information which row contains which coordinate
        self.index_dict = {coord: i for i, coord in enumerate(self._order)}

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
