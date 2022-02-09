# third party imports
import numpy as np

# local imports
from probeye.subroutines import compute_reduction_array


class SpatiotemporalExponentialCorrelationModel:
    """
    Represents a spatio-temporal correlation model with an exponential kernel. It
    contains the functionality to compute the covariance matrix for all the experiments
    given for a considered noise model.
    """

    def __init__(self, position_arrays: dict):

        # compute the distance array which might contain duplicate rows/columns
        self.distance_array = self.compute_distance_array(position_arrays)

        # the reduction array is a matrix that will be multiplied to the model error
        # in order to average the error values on indices that correspond to duplicate
        # rows/columns in the distance array
        self.reduction_array, duplicates = compute_reduction_array(self.distance_array)
        if duplicates:  # if there are duplicates, remove the respective rows/columns
            self.distance_array = np.delete(self.distance_array, duplicates, axis=0)
            self.distance_array = np.delete(self.distance_array, duplicates, axis=1)

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

    @staticmethod
    def compute_distance_array(position_array_dict: dict) -> np.ndarray:
        """
        Computes the distance (relating to space and time) of each measurement to all
        other measurements. The result is a matrix with a zero-diagonal.

        Parameters
        ----------
        position_array_dict
            The keys are the correlation variables of the noise model ('x', 'y', 'z',
            't') while the values are arrays stating the position of each measurement
            in terms of the respective correlation variable.

        Returns
        -------
        distance_array
            The distance (relating to space and time) of each measurement to all
            other measurements. Currently, this is simply the Euclidean distance where
            the time 't' is treated like another spatial component.
        """

        # each array in the values of position_array_dict has the same shape, which is
        # also going to be the shape of the distance_array
        first_key = [*position_array_dict.keys()][0]
        distance_array = np.zeros(position_array_dict[first_key].shape)

        # compute the distance currently using a simple Euclidean metric
        for v, position_array in position_array_dict.items():
            distance_array += np.power(position_array - position_array.transpose(), 2)
        distance_array = np.sqrt(distance_array)

        return distance_array

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
        cov
            The covariance matrix based on the given parameters.
        """
        # this is the correlation matrix based on an exponential model
        corr = np.exp(-self.distance_array / prms["l_corr"])
        # and finally, this is the covariance matrix assuming constant std dev.
        cov = prms["std"] ** 2 * corr
        return cov
