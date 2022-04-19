# standard library imports
from typing import TYPE_CHECKING

# third party imports
import numpy as np
from tripy.loglikelihood import chol_loglike_1D
from tripy.loglikelihood import kron_loglike_2D_tridiag
from tripy.loglikelihood import chol_loglike_2D
from tripy.loglikelihood import _loglike_multivariate_normal
from tripy.utils import inv_cov_vec_1D

# local imports
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.subroutines import len_or_one, assemble_covariance_matrix

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.forward_model import ForwardModelBase


class ScipyLikelihoodBase(GaussianLikelihoodModel):
    """
    This class serves as a parent class for all the scipy-based likelihood model classes
    that follow below. All of these classes have in common that they contain computation
    methods based on numpy-arrays and use scipy routines.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):

        super().__init__(
            prms_def,
            experiment_name,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )
        self.forward_model = forward_model

    def loglike(
        self,
        response_vector: np.ndarray,
        residual_vector: np.ndarray,
        prms: dict,
    ) -> float:
        """
        Computes and returns the log-likelihood. To be overwritten by one of the
        specific classes listed below.

        Parameters
        ----------
        response_vector
            Vector of the model responses (concatenated over output sensors).
        residual_vector
            Vector of the model residuals (concatenated over output sensors).
        prms
            Dictionary containing parameter name:value pairs.

        Returns
        -------
        ll
            A scalar value representing the evaluated log-likelihood function.
        """
        raise NotImplementedError(
            "A loglike method was not implemented for this class yet."
        )


class UncorrelatedModelError(ScipyLikelihoodBase):
    """
    This class serves as a parent class for the two scipy-based likelihood models that
    do not account for correlation. These model-classes are:
    AdditiveUncorrelatedModelError and MultiplicativeUncorrelatedModelError.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )


class CorrelatedModelError(ScipyLikelihoodBase):
    """
    This class serves as a parent class for the scipy-based likelihood model classes
    that account for some kind of correlation.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def get_correlation_vector(self, correlation_variable: str) -> np.ndarray:
        """
        Gets the correlation vector of the likelihood model and returns it. This method
        is necessary, since the correlation vector can be defined in two ways via the
        experimental data. This method checks in which way it was defined and collects
        the data correspondingly.

        Parameters
        ----------
        correlation_variable
            The name of the correlation variable. For example: 'x' and 'T_in'.

        Returns
        -------
        corr_vector
            The collected correlation vector.
        """

        if correlation_variable in self.forward_model.input_sensor_names:
            # in this case, the correlation vector is given by one of the forward
            # model's input sensors (check out the documentation for more details)
            corr_vector = self.forward_model.input_sensor_dict[correlation_variable][
                self.experiment_name
            ]
        else:
            # in this case, the correlation vector is assembled from scalars that are
            # attributed to the output sensors of the likelihood model's forward model
            corr_vector = np.zeros(self.forward_model.n_output_sensors)
            for i, output_sensor in enumerate(self.forward_model.output_sensors):
                corr_vector[i] = getattr(output_sensor, correlation_variable)

        return corr_vector

    def spatial_coordinate_array(self, correlation_variables: tuple) -> np.ndarray:
        """
        Assemble the coordinate array from the experimental data and return it.

        Parameters
        ----------
        correlation_variables
            Contains strings of the spatial coordinates. For example ('x', 'y').

        Returns
        -------
        coords_array
            The number of columns is the number of spatial dimensions, which for this
            class must be either two or three. The number of rows if the number of
            spatial points that are considered.
        """

        # allocate the array for the coordinates
        n = len(self.get_correlation_vector(correlation_variables[0]))
        d = len(correlation_variables)
        coords_array = np.zeros((n, d))

        # fill the prepared array for each spatial dimension-coordinate (in most cases
        # these correlation-variables will be 'x', 'y' or ''z)
        for i, correlation_variable in enumerate(correlation_variables):
            coords_array[:, i] = self.get_correlation_vector(correlation_variable)

        return coords_array


class CorrelatedModelError1V(CorrelatedModelError):
    """
    This class serves as a parent class for the scipy-based likelihood model classes
    that account for correlation in a single (possibly multidimensional) corr. variable.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

        # make sure there is only one correlation variable defined, and write this
        # single correlation variable to an attribute of this class
        if len(self.correlation_variables) > 1:
            raise RuntimeError(
                f"For likelihood model 'CorrelatedModelError1V' only one correlation "
                f"variable can be defined! Found self.correlation_variables"
                f" = {self.correlation_variables}."
            )


class CorrelatedModelError1D(CorrelatedModelError1V):
    """
    This class serves as a parent class for the scipy-based likelihood model classes
    that account for correlation in a single 1D-variable.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

        # make sure the only correlation variable of this class is given correctly; this
        # check is mostly to ensure type consistency
        assert isinstance(self.correlation_variables[0], str)
        self.correlation_variable = self.correlation_variables[0]

        # get the values of the only 1D correlation variable; note that these values
        # don't change because they are entirely derived from the experimental data
        self.corr_vector = self.get_correlation_vector(self.correlation_variable)


class CorrelatedModelErrorS23D(CorrelatedModelError1V):
    """
    This class serves as a parent class for the scipy-based likelihood model classes
    that account for correlation in a single spatial variable with 2 or 3 dimensions.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

        # make sure the only correlation variable of this class is given correctly; this
        # check is mostly to ensure type consistency
        assert isinstance(self.correlation_variables[0], tuple)
        self.correlation_variable = self.correlation_variables[0]

        # get the values of the spatial 2D or 3D correlation variable; note that these
        # values don't change because they are entirely derived from the exp. data;
        # these spatial coordinates will be needed for the correlation matrix
        self.coords_array = self.spatial_coordinate_array(self.correlation_variable)


class CorrelatedModelError2V(CorrelatedModelError):
    """
    This class serves as a parent class for the scipy-based likelihood model classes
    that account for correlation in two correlation variables.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )


class CorrelatedModelError1D1D(CorrelatedModelError2V):
    """
    This class serves as a parent class for the scipy-based likelihood model classes
    that account for correlation in two 1D-variables.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

        corr_var_1 = self.correlation_variables[0]
        self.corr_vector_1 = self.get_correlation_vector(corr_var_1)
        self.l_corr_1 = self.forward_model.output_sensors[0].correlated_in[corr_var_1]

        corr_var_2 = self.correlation_variables[1]
        self.corr_vector_2 = self.get_correlation_vector(corr_var_2)
        self.l_corr_2 = self.forward_model.output_sensors[1].correlated_in[corr_var_2]


class CorrelatedModelError1DS23D(CorrelatedModelError2V):
    """
    This class serves as a parent class for the scipy-based likelihood model classes
    that account for correlation in two 1D-variables.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

        # ascertain which variable is the 1D and which is the spatial one
        if isinstance(self.correlation_variables[0], tuple):
            corr_var_1D = self.correlation_variables[1]
            corr_var_23D = self.correlation_variables[0]
        else:
            corr_var_1D = self.correlation_variables[0]
            corr_var_23D = self.correlation_variables[1]

        # set attributes related to the 1D correlation variable
        self.corr_vector_1D = self.get_correlation_vector(corr_var_1D)
        self.l_corr_1D = self.forward_model.output_sensors[0].correlated_in[corr_var_1D]

        # set attributes related to the 2D/3D correlation variable
        self.corr_vector_23D = self.spatial_coordinate_array(corr_var_23D)
        self.l_corr_23D = self.forward_model.output_sensors[1].correlated_in[
            corr_var_23D
        ]


# ==================================================================================== #
#                              Additive likelihood models                              #
# ==================================================================================== #


class AdditiveUncorrelatedModelError(UncorrelatedModelError):
    """
    This is a likelihood model based on a multivariate normal distribution without any
    correlations, i.e., with a diagonal covariance matrix. Both the model error as well
    as the measurement error (if considered) are assumed to be additive.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self,
        response_vector: np.ndarray,
        residual_vector: np.ndarray,
        prms: dict,
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (SolverLikelihoodBase).
        """
        # compute the variance vector (the diagonal of the covariance matrix) or scalar
        std_model, std_meas, stds_are_scalar = self.forward_model.std_values(
            prms,
            exp_name=self.experiment_name,
            measurement_error=self.additive_measurement_error,
        )
        variance = np.power(std_model, 2)
        n = len(residual_vector)
        if std_meas is not None:
            variance += np.power(std_meas, 2)
        if stds_are_scalar:
            # in this case, 'variance' is a scalar
            ll = -n / 2 * np.log(2 * np.pi * variance)
            ll -= 0.5 / variance * np.sum(np.square(residual_vector))
        else:
            # in this case, 'variance' is a  (non-constant) vector
            ll = -0.5 * (n * np.log(2 * np.pi) + np.sum(np.log(variance)))
            ll -= 0.5 * np.sum(np.square(residual_vector) / variance)
        return float(ll)


class AdditiveCorrelatedModelError1D(CorrelatedModelError1D):
    """
    This is a likelihood model class based on a multivariate normal distribution that
    includes correlation effects in a single 1D-variable. Both the model error as well
    as the measurement error (if considered) are assumed to be additive.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self,
        response_vector: np.ndarray,
        residual_vector: np.ndarray,
        prms: dict,
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (SolverLikelihoodBase).
        """

        # get the standard deviations (vectors or scalars) and the correlation length;
        # note that in this case it does not matter if the std-values are vectors or
        # scalars since both formats are compatible with 'chol_loglike_1D'
        std_model, std_meas, _ = self.forward_model.std_values(
            prms,
            exp_name=self.experiment_name,
            corr_var=self.correlation_variable,
            measurement_error=self.additive_measurement_error,
        )
        l_corr = prms["l_corr"]

        # efficient log-likelihood evaluation via tripy
        ll = chol_loglike_1D(
            residual_vector, self.corr_vector, l_corr, std_model, std_meas
        )
        return ll


class AdditiveCorrelatedModelErrorS23D(CorrelatedModelErrorS23D):
    """
    This is a likelihood model class based on a multivariate normal distribution that
    includes correlation effects in two or three spatial variables. Both the model error
    as well as the measurement error (if considered) are assumed to be additive.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self,
        response_vector: np.ndarray,
        residual_vector: np.ndarray,
        prms: dict,
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (SolverLikelihoodBase).
        """

        # get the standard deviations (vectors or scalars) and the correlation length;
        # note that in this case it does not matter if the std-values are vectors or
        # scalars since both formats are compatible with 'assemble_covariance_matrix';
        # also note that for this likelihood model 'self.correlation_variable' is a
        # tuple with 2 or 3 elements, like ('x', 'y') or ('x', 'y', 'z')
        std_model, std_meas, _ = self.forward_model.std_values(
            prms,
            exp_name=self.experiment_name,
            corr_var=self.correlation_variable[0],
            measurement_error=self.additive_measurement_error,
        )
        l_corr = prms["l_corr"]

        # assemble the covariance matrix
        cov_matrix = assemble_covariance_matrix(
            self.coords_array, std_model, std_meas, l_corr
        )

        # evaluate log-likelihood (no efficient algorithm available in this case)
        return _loglike_multivariate_normal(residual_vector, cov_matrix)


class AdditiveCorrelatedModelError1D1D(CorrelatedModelError1D1D):
    """
    This is a likelihood model class based on a multivariate normal distribution that
    includes correlation effects two 1D correlation variables. Both the model error as
    well as the measurement error (if considered) are assumed to be additive.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self,
        response_vector: np.ndarray,
        residual_vector: np.ndarray,
        prms: dict,
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (SolverLikelihoodBase).
        """

        # compute the model residuals via a method from the parent class
        res_array = residual_vector.reshape((self.forward_model.n_output_sensors, -1))

        # compute the vectors of standard deviations and get the correlation length
        corr_var_1 = self.forward_model.output_sensors[0].correlation_variables[0]
        std_model, std_meas, stds_are_scalar = self.forward_model.std_values(
            prms,
            exp_name=self.experiment_name,
            corr_var=corr_var_1,
            measurement_error=self.additive_measurement_error,
        )
        l_corr_1 = prms[self.l_corr_1]
        l_corr_2 = prms[self.l_corr_2]

        if stds_are_scalar:
            # if both std_model and std_meas are scalars or None, we can use the
            # following fast tripy-method to evaluate the likelihood
            ll = kron_loglike_2D_tridiag(
                res_array,
                self.corr_vector_1,
                self.corr_vector_2,
                l_corr_1,
                std_model,
                l_corr_2,
                1,
                std_meas,
            )

        else:
            # in this case, std_model and/or std_meas are vectors, so we have to
            # refrain to a slower tripy-method
            d0_1, d1_1 = inv_cov_vec_1D(self.corr_vector_1, l_corr_1, std_model)
            d0_2, d1_2 = inv_cov_vec_1D(self.corr_vector_2, l_corr_2, 1.0)
            ll = chol_loglike_2D(res_array, [d0_1, d1_1], [d0_2, d1_2], std_meas)

        return ll


class AdditiveCorrelatedModelError1DS23D(CorrelatedModelError1DS23D):
    """
    This is a likelihood model based on a multivariate normal distribution with that
    includes correlation effects in two correlation variables, one general 1D-variable
    and one 2- or 3-dimensional spatial variable. Both the model error as well as the
    measurement error (if considered) are assumed to be additive.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self,
        response_vector: np.ndarray,
        residual_vector: np.ndarray,
        prms: dict,
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (SolverLikelihoodBase).
        """

        # compute the model residuals via a method from the parent class
        res_array = residual_vector.reshape((self.forward_model.n_output_sensors, -1))

        # compute the vectors of standard deviations and get the correlation length
        corr_var_1 = self.forward_model.output_sensors[0].correlation_variables[0]
        std_model, std_meas, stds_are_scalar = self.forward_model.std_values(
            prms,
            exp_name=self.experiment_name,
            corr_var=corr_var_1,
            measurement_error=self.additive_measurement_error,
        )
        l_corr_1D = prms[self.l_corr_1D]
        l_corr_23D = prms[self.l_corr_23D]

        # assemble the covariance matrix
        spatial_cov_matrix = assemble_covariance_matrix(
            self.corr_vector_23D, std_model, std_meas, l_corr_23D
        )

        # get the main diagonal and off-diagonal of the time covariance matrix inverse
        d0_t, d1_t = inv_cov_vec_1D(self.corr_vector_1D, l_corr_1D, 1.0)

        # efficient log-likelihood evaluation via tripy
        ll = chol_loglike_2D(res_array, spatial_cov_matrix, [d0_t, d1_t], std_meas)

        return ll


# ==================================================================================== #
#                           Multiplicative likelihood models                           #
# ==================================================================================== #


class MultiplicativeUncorrelatedModelError(UncorrelatedModelError):
    """
    This is a likelihood model based on a multivariate normal distribution without any
    correlations, i.e., with a diagonal covariance matrix. The model error is assumed to
    be multiplicative while the measurement error (if considered) is assumed additive.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self,
        response_vector: np.ndarray,
        residual_vector: np.ndarray,
        prms: dict,
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (SolverLikelihoodBase).
        """
        # process the standard deviations
        std_model, std_meas, _ = self.forward_model.std_values(
            prms,
            exp_name=self.experiment_name,
            measurement_error=self.additive_measurement_error,
        )

        # compute the covariance matrix depending on the measurement error; note that
        # without correlation (as it is the case here) the covariance matrix is diagonal
        # and can hence be represented by a single vector; this is the case in the
        # computations below, i.e., cov_mtx is a vector; note that both of the following
        # operations work irrespective of std_model/std_meas being scalars or vectors
        cov_mtx = np.power(response_vector * std_model, 2)
        if std_meas is not None:
            cov_mtx += np.power(std_meas, 2)

        # finally, evaluate the log-likelihood
        n = len_or_one(residual_vector)
        inv_cov_mtx = 1 / cov_mtx
        log_det_cov_mtx = np.sum(np.log(cov_mtx))
        ll = -0.5 * (n * np.log(2.0 * np.pi) + log_det_cov_mtx)
        ll += -0.5 * np.sum(np.power(residual_vector, 2) * inv_cov_mtx)
        return ll


class MultiplicativeCorrelatedModelError1D(CorrelatedModelError1D):
    """
    This is a likelihood model class based on a multivariate normal distribution that
    includes correlation effects in a single 1D-variable. The model error is assumed to
    be multiplicative while the measurement error (if considered) is additive.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self,
        response_vector: np.ndarray,
        residual_vector: np.ndarray,
        prms: dict,
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (SolverLikelihoodBase).
        """

        # get the standard deviations (vectors or scalars) and the correlation length;
        # note that in this case it does not matter if the std-values are vectors or
        # scalars since both formats are compatible with 'chol_loglike_1D'
        std_model, std_meas, _ = self.forward_model.std_values(
            prms,
            exp_name=self.experiment_name,
            measurement_error=self.additive_measurement_error,
        )
        l_corr = prms["l_corr"]

        # efficient log-likelihood evaluation via tripy
        ll = chol_loglike_1D(
            residual_vector,
            self.corr_vector,
            l_corr,
            std_model,
            std_meas,
            y_model=response_vector,
        )
        return ll


class MultiplicativeCorrelatedModelErrorS23D(CorrelatedModelErrorS23D):
    """
    This is a likelihood model class based on a multivariate normal distribution that
    includes correlation effects in two or three spatial variables. The model error is
    assumed to be multiplicative while the measurement error (if considered) is assumed
    to be additive.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self,
        response_vector: np.ndarray,
        residual_vector: np.ndarray,
        prms: dict,
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (SolverLikelihoodBase).
        """

        # get the standard deviations (vectors or scalars) and the correlation length;
        # note that in this case it does not matter if the std-values are vectors or
        # scalars since both formats are compatible with 'assemble_covariance_matrix';
        # also note that for this likelihood model 'self.correlation_variable' is a
        # tuple with 2 or 3 elements, like ('x', 'y') or ('x', 'y', 'z')
        std_model, std_meas, _ = self.forward_model.std_values(
            prms,
            exp_name=self.experiment_name,
            corr_var=self.correlation_variable[0],
            measurement_error=self.additive_measurement_error,
        )
        l_corr = prms["l_corr"]

        # assemble the covariance matrix
        cov_matrix = assemble_covariance_matrix(
            self.coords_array, std_model, std_meas, l_corr, y_model=response_vector
        )

        # evaluate log-likelihood (no efficient algorithm available in this case)
        return _loglike_multivariate_normal(residual_vector, cov_matrix)


class MultiplicativeCorrelatedModelError1D1D(CorrelatedModelError1D1D):
    """
    This is a likelihood model class based on a multivariate normal distribution that
    includes correlation effects two 1D correlation variables. The model error is
    multiplicative while the measurement error (if considered) is additive.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self,
        response_vector: np.ndarray,
        residual_vector: np.ndarray,
        prms: dict,
        worst_value: float = -np.infty,
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (SolverLikelihoodBase).
        """

        # compute the model residuals via a method from the parent class
        residuals_array = residual_vector.reshape(
            (self.forward_model.n_output_sensors, -1)
        )
        response_array = response_vector.reshape(
            (self.forward_model.n_output_sensors, -1)
        )

        # compute the vectors of standard deviations and get the correlation length
        corr_var_1 = self.forward_model.output_sensors[0].correlation_variables[0]
        std_model, std_meas, _ = self.forward_model.std_values(
            prms,
            exp_name=self.experiment_name,
            corr_var=corr_var_1,
            measurement_error=self.additive_measurement_error,
        )
        l_corr_1 = prms[self.l_corr_1]
        l_corr_2 = prms[self.l_corr_2]

        # independent of the std-values being scalar- or vector-valued, we have to use
        # the tripy chol_loglike_2D method for the likelihood evaluation
        d0_1, d1_1 = inv_cov_vec_1D(self.corr_vector_1, l_corr_1, std_model)
        d0_2, d1_2 = inv_cov_vec_1D(self.corr_vector_2, l_corr_2, 1.0)
        ll = chol_loglike_2D(
            residuals_array,
            [d0_1, d1_1],
            [d0_2, d1_2],
            std_meas,
            y_model=response_array,
        )

        return ll


class MultiplicativeCorrelatedModelError1DS23D(CorrelatedModelError1DS23D):
    """
    This is a likelihood model based on a multivariate normal distribution with that
    includes correlation effects in two correlation variables, one general 1D-variable
    and one 2- or 3-dimensional spatial variable. The model error is multiplicative
    while the measurement error (if considered) is assumed to be additive.

    Parameters
    ----------
    prms_def
        Global parameter names defining which parameters are used by the likelihood
        model. For example prms_def = ['sigma', 'l_corr'] or prms_def='sigma_1'.
    experiment_name
        The name of the experiment the likelihood model refers to. Note that each
        likelihood model refers to exactly one experiment.
    forward_model
        The forward model the error of which the likelihood model assesses.
    additive_measurement_error
        If True, next to the model error, a normal, zero-mean i.i.d. measurement error
        is assumed to be present.
    correlation_variables
        Defines the correlation variables, i.e. variables, the forward model prediction
        error is assumed to be correlated with.
    correlation_model
        Defines the correlation function to be used in case correlation is considered
        (which is the case, when correlation_variables is a non-empty string).
        Currently, there is only one option 'exp' which represents an exponential model.
        In the future, more options should be added.
    name
        Unique name of the likelihood model. This name is None, if the user does not
        specify it when adding the likelihood model to the problem. It is then named
        automatically before starting the inference engine.
    """

    def __init__(
        self,
        prms_def: dict,
        experiment_name: str,
        forward_model: "ForwardModelBase",
        additive_measurement_error: bool,
        correlation_variables: list,
        correlation_model: str,
        name: str,
    ):
        super().__init__(
            prms_def,
            experiment_name,
            forward_model,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            name=name,
        )

    def loglike(
        self,
        response_vector: np.ndarray,
        residual_vector: np.ndarray,
        prms: dict,
    ) -> float:
        """
        Computes the log-likelihood of this model. For more information, check out the
        doc-string of the parent class (SolverLikelihoodBase).
        """

        # compute the model residuals via a method from the parent class
        residual_array = residual_vector.reshape(
            (self.forward_model.n_output_sensors, -1)
        )
        response_array = response_vector.reshape(
            (self.forward_model.n_output_sensors, -1)
        )

        # compute the vectors of standard deviations and get the correlation length
        corr_var_1 = self.forward_model.output_sensors[0].correlation_variables[0]
        std_model, std_meas, stds_are_scalar = self.forward_model.std_values(
            prms,
            exp_name=self.experiment_name,
            corr_var=corr_var_1,
            measurement_error=self.additive_measurement_error,
        )
        l_corr_1D = prms[self.l_corr_1D]
        l_corr_23D = prms[self.l_corr_23D]

        # assemble the covariance matrix
        spatial_cov_matrix = assemble_covariance_matrix(
            self.corr_vector_23D, std_model, std_meas, l_corr_23D
        )

        # get the main diagonal and off-diagonal of the time covariance matrix inverse
        d0_t, d1_t = inv_cov_vec_1D(self.corr_vector_1D, l_corr_1D, 1.0)

        # efficient log-likelihood evaluation via tripy
        ll = chol_loglike_2D(
            residual_array,
            spatial_cov_matrix,
            [d0_t, d1_t],
            std_meas,
            y_model=response_array,
        )

        return ll


def translate_likelihood_model(lm_def: GaussianLikelihoodModel) -> ScipyLikelihoodBase:
    """
    Translates a given instance of GaussianLikelihoodModel (which is essentially just a
    description of the likelihood model without any computing-methods) to a specific
    likelihood model object which does contain SciPy-based computing-methods.

    Parameters
    ----------
    lm_def
        An instance of GaussianLikelihoodModel which contains general information on the
        likelihood model but no computing-methods.
    Returns
    -------
    likelihood_computer
        An instance of a specific likelihood model class with computing capabilities.
        Examples for such classes are given above in this file.
    """

    # likelihood model selection based on the flags given in the likelihood definition
    prefix = "Add" if lm_def.additive_model_error else "Mul"
    if not lm_def.considers_correlation:
        l_class = f"{prefix}_Uncorrelated"
    else:
        if lm_def.n_correlation_variables == 1:
            if lm_def.has_S23D_correlation_variable:
                l_class = f"{prefix}_Correlated_S23D"
            else:
                l_class = f"{prefix}_Correlated_1D"
        else:
            if lm_def.has_S23D_correlation_variable:
                l_class = f"{prefix}_Correlated_1DS23D"
            else:
                l_class = f"{prefix}_Correlated_1D1D"

    # this dict allows to map an assigned string from the if-cases above to a specific
    # likelihood model defined in this file; the class is not assigned directly to the
    # variable l_class from the if-clauses above to avoid that l_class can have multiple
    # different types (which leads to issues during type-checking)
    class_dict = {
        "Add_Uncorrelated": AdditiveUncorrelatedModelError,
        "Add_Correlated_1D": AdditiveCorrelatedModelError1D,
        "Add_Correlated_S23D": AdditiveCorrelatedModelErrorS23D,
        "Add_Correlated_1D1D": AdditiveCorrelatedModelError1D1D,
        "Add_Correlated_1DS23D": AdditiveCorrelatedModelError1DS23D,
        "Mul_Uncorrelated": MultiplicativeUncorrelatedModelError,
        "Mul_Correlated_1D": MultiplicativeCorrelatedModelError1D,
        "Mul_Correlated_S23D": MultiplicativeCorrelatedModelErrorS23D,
        "Mul_Correlated_1D1D": MultiplicativeCorrelatedModelError1D1D,
        "Mul_Correlated_1DS23D": MultiplicativeCorrelatedModelError1DS23D,
    }

    # this is where the translation happens
    likelihood_computer = class_dict[l_class](
        lm_def.prms_def,
        lm_def.experiment_name,
        forward_model=lm_def.forward_model,
        additive_measurement_error=lm_def.additive_measurement_error,
        correlation_variables=lm_def.correlation_variables,
        correlation_model=lm_def.correlation_model,
        name=lm_def.name,
    )

    return likelihood_computer
