# standard library imports
import logging

# third party imports
import numpy as np

# prefix logging output with the package's name
logger = logging.getLogger(__name__)


def transformation_SQ(locations, correlation_length, tol=1e-8):
    """
    Transformation matrix based on a squared exponential correlation matrix.

    Parameters
    ----------
    locations : list, tuple or numpy-array
        A (1D) vector with N scalar 'locations', e.g. [1.2, 4.7, 2.1]
    correlation_length : float
        A number controlling the extend of the correlation between points.

    Returns
    -------
    2D numpy array
        An (N x M) transformation matrix with M<=N.
    """
    return transformation_from_correlation(
        squared_exponential(locations, correlation_length), tol
    )


def squared_exponential(locations, correlation_length):
    """
    Builds a dense correlation matrix assuming the 'locations' are correlated by
    'correlation_length' (this is essentially a scaling-hyperparameter). Note
    that the reference to distance-like names should not prevent you from
    putting in 'times' and something like a 'correlation_duration'.

    Parameters
    ----------
    locations : list, tuple or numpy-array
        A (1D) vector with scalar 'locations', e.g. [1.2, 4.7, 2.1]
    correlation_length : float
        A number controlling the extend of the correlation between points.

    Returns
    -------
        numpy array
        The correlation matrix of shape (len(locations), len(locations))
    """
    # locations must be a 1D vector
    loc = np.atleast_1d(locations)
    assert len(loc.shape) == 1
    # build matrix with len(loc) rows containing loc
    c0 = np.repeat([loc], len(loc), axis=0)
    r = c0 - c0.T  # these are the distances from each location to all others
    # compare https://en.wikipedia.org/wiki/Covariance_function
    return np.exp(-r * r / (2.0 * correlation_length * correlation_length))


def transformation_from_correlation(correlation, tol=1e-8):
    """
    Decompose the covariance matrix into its principal components
    Only keep the eigenvalues e with e > tol * largest eigenvalue

    Return the diagonal entries (representing the squares of the std_dev
    of independent random variables) and the corresponding eigenvectors  

    The full (correlated) sample vector X is then given by
    X = sum_i Phi_i * X_red,i with X_red,i being normal distributed with 
    zero mean and sigma2 given by the eigenvalues of the covariance matrix and
    Phi_i the corresponding eigenvalues.

    Parameters
    ----------
    correlation : 2D numpy array
        The correlation matrix, a square, symmetric, real (N x N)-matrix.
    tol : float, optional
        Tolerance value for discarding small eigenvalues.

    Returns
    -------
    transformation : 2D numpy array
        An (N x M) transformation matrix with M<=N.
    """

    # since the correlation matrix is symmetric, we can use np.linalg.eigh
    eigenvalues, eigen_vectors = np.linalg.eigh(correlation)
    # note that the last element of eigenvalues is the largest eigenvalue
    threshold = tol * eigenvalues[-1]
    # remove eigenvalues and corresponding eigenvectors with eigenvalues smaller
    # than the previously computed threshold
    reduced_eigenvalues = eigenvalues[eigenvalues > threshold]
    reduced_eigenvectors = eigen_vectors[:, eigenvalues > threshold]
    # divide each eigenvector element-wise by the square root of its eigenvalue
    transformation = np.divide(reduced_eigenvectors,
                               np.sqrt(reduced_eigenvalues))

    # print out some relevant information for debugging
    logger.info(f"Shape of transformation matrix:\n{transformation.shape}")
    logger.debug(f"Full set of eigenvalues (columns):\n{eigenvalues}")
    logger.debug(f"Threshold for reduction:\n{threshold}")
    logger.debug(f"Reduced set of eigenvalues (columns):" +
                 f"\n{reduced_eigenvalues}")

    return transformation
