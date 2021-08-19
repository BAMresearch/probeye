# third party imports
import numpy as np

# local imports
from probeye.subroutines import len_or_one


def delta_x(x0, delta=None):
    """
    Returns a good choice for the step size for numeric differentiation around
    x0 when using a simple two-point approximation, see also the value h in
    https://en.wikipedia.org/wiki/Numerical_differentiation.

    Parameters
    ----------
    x0 : float
        Point where the derivative should be evaluated.
    delta : float, optional
        This parameter can be used when a specific fixed value should be
        returned; might be used for debugging.
        
    Returns
    -------
    dx : float
        Computed step size for numeric differentiation.
    """
    # use the delta-value if delta is specified
    if delta is not None:
        return delta
    eps = 1e-7  # approx sqrt(machine precision)
    dx = x0 * eps + eps  #TODO: clarify if the '+ eps' part is correct
    if dx == 0:
        dx = eps
    return dx


def d_model_error_d_named_parameter(model_error, prm_name):
    """
    Computes the derivative of the model_error with respect to the parameter
    prm_name.

    Parameters
    ----------
    model_error :  object
        An object that has an attribute 'parameter_list' and a __call__()
        method without arguments that returns a dictionary with elements like
        {key : numpy_vector of length N} stating the model errors of sensors.
    prm_name :  string
        The name of the parameter with respect to which the derivative of the
        model error should be computed.
        
    Returns
    -------
        jac : dict
        A dictionary with sensor names as keys and their derivatives as values,
        e.g. {'exp1' : {ForceSensor1 : np.array([-0.3])}}.
    """
    # distinguish between scalar and vector-valued parameters
    if len_or_one(model_error.parameter_list[prm_name]) == 1:
        jac = d_model_error_d_scalar_parameter(model_error, prm_name)
    else:
        jac = d_model_error_d_vector_parameter(model_error, prm_name)
    return jac


def d_model_error_d_scalar_parameter(model_error, prm_name):
    """
    Calculates the derivative of 'model_error' w.r.t. the parameter with name
    'prm_name'. This method assumes that the parameter with name 'prm_name' is
    a scalar parameter and not a vector-valued one.

    Parameters
    ----------
    model_error :  object
        An object that has an attribute 'parameter_list' and a __call__()
        method without arguments that returns a dictionary with elements like
        {key : numpy_vector of length N} stating the model errors of sensors.
    prm_name :  string
        The name of the parameter with respect to which the derivative of the
        model error should be computed.

    Returns
    -------
        jac : dict
        A dictionary with sensor names as keys and their derivatives as values,
        e.g. {'exp1' : {ForceSensor1 : np.array([-0.3])}}.
    """

    # extract the parameter value and compute a good step size for it
    prm0 = model_error.parameter_list[prm_name]
    dx = delta_x(prm0)

    # evaluate the model error on the 'left'
    model_error.parameter_list[prm_name] = prm0 - dx
    me0 = model_error()
    # evaluate the model error on the 'right'
    model_error.parameter_list[prm_name] = prm0 + dx
    me1 = model_error()
    # reset the parameter to its original value
    model_error.parameter_list[prm_name] = prm0

    # compute the numeric derivative of all model errors with respect to the
    # considered parameter and write the results to a dictionary
    jac = dict()
    for key in me0:
        # compute symmetric difference approximation
        jac[key] = (me1[key] - me0[key]) / (2 * dx)

    return jac


def d_model_error_d_vector_parameter(model_error, prm_name):
    """
    Calculates the derivative of 'model_error' w.r.t. the parameter with name
    'prm_name'. This method assumes that the parameter with name 'prm_name' is
    a vector-valued parameter and not a scalar one.

    Parameters
    ----------
    model_error :  object
        An object that has an attribute 'parameter_list' and a __call__()
        method without arguments that returns a dictionary with elements like
        {key : numpy_vector of length N} stating the model errors of sensors.
    prm_name :  string
        The name of the parameter with respect to which the derivative of the
        model error should be computed.

    Returns
    -------
        jac : dict
        A dictionary with sensor names as keys and their derivatives as values,
        e.g. {'exp1' : {PositionSensor1 : np.array([-0.3, 0.2, 0.1])}}.
    """

    # extract the parameter value and determine its dimension
    prm0 = np.copy(model_error.parameter_list[prm_name])
    dim = len(prm0)

    # approximate the partial derivatives by looping over the parameter's
    # elements (remember, it's a vector-valued one)
    jac = dict()
    for row in range(dim):
        # compute a good step size for the finite difference scheme
        dx = delta_x(prm0[row])
        # evaluate the model error on the 'left'
        model_error.parameter_list[prm_name][row] = prm0[row] - dx
        me0 = model_error()
        # evaluate the model error on the 'right'
        model_error.parameter_list[prm_name][row] = prm0[row] + dx
        me1 = model_error()
        # reset the parameter element to its original value
        model_error.parameter_list[prm_name][row] = prm0[row]

        # compute the numeric derivative of all model errors with respect to the
        # considered parameter element and write the results to a dictionary
        for key in me0:
            if key not in jac:  # initialize the derivative vector
                jac[key] = np.empty((len(me0[key]), dim))
            # compute symmetric difference approximation
            jac[key][:, row] = (me1[key] - me0[key]) / (2 * dx)

    return jac
