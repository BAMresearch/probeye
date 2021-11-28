# standard library
from typing import Union, Optional, Tuple

# third party imports
import numpy as np

# local imports
from probeye.subroutines import process_spatial_coordinates


class SpatialExponentialCorrelationModel:
    """
    Represents a spatial correlation model with an exponential kernel. It contains the
    functionality to compute the covariance matrix over a static (i.e. constant for all
    experiments) grid of coordinates in 1D, 2D or 3D.
    """

    def __init__(
        self,
        x: Union[int, float, np.ndarray, None] = None,
        y: Union[int, float, np.ndarray, None] = None,
        z: Union[int, float, np.ndarray, None] = None,
        coords: Optional[np.ndarray] = None,
        order: Tuple = ("x", "y", "z"),
    ):
        """
        Parameter
        ---------
        x
            Positional x-coordinate. When given, coords must be None.
        y
            Positional y-coordinate. When given, coords must be None.
        z
            Positional z-coordinate. When given, coords must be None.
        coords
            Some or all of the coordinates x, y, z concatenated as an array. Each row
            corresponds to one coordinate. For example, row 1 might contain all
            x-coordinates. Which row corresponds to which coordinate is defined via the
            order-argument. When the coords-argument is given, all 3 arguments x, y and
            z must be None.
        order
            Only relevant when coords is given. Defines which row in coords corresponds
            to which coordinate. For example, order=('x', 'y', 'z') means that the 1st
            row are x-coordinates, the 2nd row are y-coords and the 3rd row are the
            z-coordinates.
        """
        # translate the spatial input to a coords-array
        self.coords, self._order = process_spatial_coordinates(
            x=x, y=y, z=z, coords=coords, order=order
        )
        self.n_coords, self.n = self.coords.shape

        # on position (i, j) in self.distance array will be denoted the distance between
        # point i with coords[i, :] and point j with coords[j, :]
        distance_array = np.zeros((self.n, self.n))
        for i in range(self.n_coords):
            v = self.coords[i, :]
            v_in_columns = np.tile(v.reshape((self.n, -1)), self.n)
            v_in_rows = v_in_columns.transpose()  # each row is v
            distance_array += np.square(v_in_columns - v_in_rows)
        self.distance_array = np.sqrt(distance_array)

    @staticmethod
    def check_prms(prms: dict) -> bool:
        """
        Checks if the numeric values provided via prms are valid for computing the
        covariance matrix of this model.

        Parameters
        ----------
        prms
            Contains the names of the correlation model's parameters as keys and the
            corresponding numeric values as values.

        Returns
        -------
            True, when all values are valid. False otherwise.
        """
        # if no correlation is defined in the noise model, l_corr will not be provided
        # in the input for the log-likelihood contribution
        if "l_corr" in prms:
            if prms["l_corr"] <= 0:
                return False
        # sigma, however, must always be passed
        if prms["std"] <= 0:
            return False
        return True

    def __call__(self, prms: dict) -> np.ndarray:
        """
        Returns the covariance matrix based on the correlation model.

        Parameters
        ----------
        prms
            Contains the names of the correlation model's parameters as keys and the
            corresponding numeric values as values.

        Returns
        -------
            The covariance matrix based on the given parameters. The shape of
            this array is (self.n, self.n).
        """
        corr = np.exp(-self.distance_array / prms["l_corr"])
        return prms["std"] ** 2 * corr
