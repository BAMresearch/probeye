# third party imports
import numpy as np

# local imports
from probeye.subroutines import process_spatiotemporal_coordinates


class SpatiotemporalExponentialCorrelationModel:
    """
    Represents a spatial correlation model with an exponential kernel. It
    contains the functionality to compute the covariance matrix over a static
    (i.e. constant for all experiments) grid of coordinates in 1D, 2D or 3D.
    """

    def __init__(self, position_arrays):

        self.distance_array = self.compute_distance_array(position_arrays)

    @staticmethod
    def check_prms(prms):
        """
        Checks if the numeric values provided via prms are valid for computing
        the covariance matrix of this model.

        Parameters
        ----------
        prms : dict
            Contains the names of the correlation model's parameters as keys
            and the corresponding numeric values as values.

        Returns
        -------
        bool
            True, when all values are valid. False otherwise.
        """
        # if no correlation is defined in the noise model, l_corr will not be
        # provided in the input for the log-likelihood contribution
        if 'l_corr' in prms:
            if prms['l_corr'] <= 0:
                return False
        # sigma, however, must always be passed
        if prms['std'] <= 0:
            return False
        return True

    def compute_distance_array(self, position_arrays):
        """
        Computes the distance (in terms space and time) of each measurement to
        all other measurements. The result is a matrix with a zero-diagonal.
        """

        first_key = [*position_arrays.keys()][0]
        distance_array = np.zeros(position_arrays[first_key].shape)
        for v, position_array in position_arrays.items():
            distance_array +=\
                np.power(position_array - position_array.transpose(), 2)
        distance_array = np.sqrt(distance_array)

        return distance_array

    def __call__(self, prms):
        """
        Returns the covariance matrix based on the correlation model.

        Parameters
        ----------
        prms : dict
            Contains the names of the correlation model's parameters as keys
            and the corresponding numeric values as values.
        corr_data : dict, None, optional
            Measurements of a single experiment, via which the correlation
            between two model errors (i.e., model response vs. measurement) is
            computed. The keys are the sensor names, and the values the sensor
            values. Note that corr_data only contains those sensor values that
            can generally change from experiment to experiment. Sensor values
            that are fix over all experiments (e.g., the position of a strain
            gauge) should be passed via the corresponding arguments during the
            initialization of this correlation model (i.e., in __init__).

        Returns
        -------
        cov : numpy.ndarray
            The covariance matrix based on the given parameters.
        """
        # this is the correlation matrix based on an exponential model
        corr = np.exp(-self.distance_array / prms['l_corr'])
        # and finally, this is the covariance matrix assuming constant std dev.
        cov = prms['std'] ** 2 * corr
        return cov
