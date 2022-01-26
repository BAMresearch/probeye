# standard library imports
from typing import Union, List, Optional, TYPE_CHECKING

# third party imports
import numpy as np
from tripy.loglikelihood import chol_loglike_1D
from tripy.loglikelihood import kron_loglike_2D_tridiag
from tripy.utils import correlation_matrix
from tripy.utils import correlation_function

# local imports
from probeye.definition.likelihood_model import GaussianLikelihoodModel
from probeye.subroutines import incrementalize

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.sensor import Sensor


class ScipyLikelihoodBase(GaussianLikelihoodModel):
    """
    This class serves as a parent class for all the scipy-based likelihood models that
    follow below. It contains computational methods based on numpy/scipy. In particular,
    these methods use numpy.ndarrays (in contrast to torch.Tensor objects as in pyro).
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str], None] = None,
        problem_experiments: Optional[dict] = None,
        additive_measurement_error: bool = False,
        correlation_variables: str = "",
        correlation_model: str = "exp",
        correlation_dict: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        """
        For a detailed explanation of the input arguments check out the docstring given
        in probeye/definition/likelihood_models.py:GaussianLikelihoodModel. The only
        additional argument is 'problem_experiments' which is simply a pointer to
        InferenceProblem._experiments (a dictionary of all the problem's experiments).
        """
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            correlation_dict=correlation_dict,
            name=name,
        )

        # problem_experiments is an attribute not set by the parent-class because when
        # objects of the parent class are created not all experiments need to have been
        # defined yet
        self.problem_experiments = problem_experiments  # type: ignore


class AdditiveUncorrelatedModelError(GaussianLikelihoodModel):
    """
    This is a likelihood model based on a multivariate normal distribution without any
    correlations, i.e., with a diagonal covariance matrix. Both the model error as well
    as the measurement error (if considered) are assumed to be additive.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str], None] = None,
        problem_experiments: Optional[dict] = None,
        additive_measurement_error: bool = False,
        correlation_variables: str = "",
        correlation_model: str = "exp",
        correlation_dict: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        """
        For a detailed explanation of the input arguments check out the docstring given
        in probeye/definition/likelihood_models.py:GaussianLikelihoodModel. The only
        additional argument is 'problem_experiments' which is simply a pointer to
        InferenceProblem._experiments (a dictionary of all the problem's experiments).
        """

        # initialize the super-class (GaussianLikelihoodModel) based on the given input
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            correlation_dict=correlation_dict,
            name=name,
        )

        # problem_experiments is an attribute not set by the parent-class
        self.problem_experiments = problem_experiments  # type: ignore

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log(likelihood) of this model.
        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.
        prms
            Dictionary containing parameter name:value pairs.
        worst_value
            This value is returned when this method does not result in a numeric value.
            This might happen for example when the given parameters are not valid (for
            example in case of a negative standard deviation). The returned value in
            such cases should represent the worst possible value of the contribution.
        Returns
        -------
        ll
            A scalar value representing the evaluated log-likelihood function.
        """
        # compute the model error; note that this mode has exactly one sensor
        res_vector = self.residuals_vector(model_response_dict)
        # process the standard deviation(s)
        std = prms["std_model"]
        if std <= 0:
            return worst_value
        if self.additive_measurement_error:
            if prms["std_measurement"] <= 0:
                return worst_value
            # this is the formula for the std. dev. of the sum of two normal dist.
            std = np.sqrt(prms["std_model"] ** 2 + prms["std_measurement"] ** 2)
        # the precision 'prec' is defined as the inverse of the variance, hence
        # prec = 1 / sigma**2 where sigma denotes the standard deviation
        prec = 1.0 / std ** 2.0
        # evaluate the Gaussian log-PDF with zero mean and a variance of 1/prec for
        # each error term and sum them up
        ll = -len(res_vector) / 2 * np.log(2 * np.pi / prec)
        ll -= 0.5 * prec * np.sum(np.square(res_vector))
        return ll


class AdditiveCorrelatedModelError1D(GaussianLikelihoodModel):
    """
    This is a likelihood model based on a multivariate normal distribution with that
    includes correlation effects in time or in one spatial coordinate. Both the model
    error as well as the measurement error (if considered) are assumed to be additive.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str], None] = None,
        problem_experiments: Optional[dict] = None,
        additive_measurement_error: bool = False,
        correlation_variables: str = "",
        correlation_model: str = "exp",
        correlation_dict: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        """
        For a detailed explanation of the input arguments check out the docstring given
        in probeye/definition/likelihood_models.py:GaussianLikelihoodModel. The only
        additional argument is 'problem_experiments' which is simply a pointer to
        InferenceProblem._experiments (a dictionary of all the problem's experiments).
        """

        # initialize the super-class (GaussianLikelihoodModel) based on the given input
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            correlation_dict=correlation_dict,
            name=name,
        )

        # problem_experiments is an attribute not set by the parent-class
        self.problem_experiments = problem_experiments  # type: ignore

        # extract the values of the correlation variable
        coords = self.coordinate_array(correlation_variables)
        self.coords, self.f, self.sorted_coords = incrementalize(coords)

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log(likelihood) of this model.
        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.
        prms
            Dictionary containing parameter name:value pairs.
        worst_value
            This value is returned when this method does not result in a numeric value.
            This might happen for example when the given parameters are not valid (for
            example in case of a negative standard deviation). The returned value in
            such cases should represent the worst possible value of the contribution.
        Returns
        -------
        ll
            A scalar value representing the evaluated log-likelihood function.
        """

        # compute the model residuals via a method from the parent class
        res_vector = self.residuals_vector(model_response_dict)
        if not self.sorted_coords:
            res_vector = self.f(res_vector)
        ones = np.ones(len(res_vector))

        # parameters for the model prediction error
        std_model = prms["std_model"]
        l_corr = prms["l_corr"]
        if (std_model <= 0) or (l_corr <= 0):
            return worst_value
        std_model *= ones

        # parameter for the measurement error
        if self.additive_measurement_error:
            std_meas = prms["std_measurement"]
            if std_meas <= 0:
                return worst_value
            std_meas *= ones
        else:
            std_meas = None

        # efficient log-likelihood evaluation via tripy
        ll = chol_loglike_1D(self.coords, res_vector, l_corr, std_model, std_meas)
        return ll


class AdditiveSpaceCorrelatedModelError2D3D(GaussianLikelihoodModel):
    """
    This is a likelihood model based on a multivariate normal distribution with that
    includes correlation effects in more than one spatial coordinate. Time correlation
    effects cannot be considered. Both the model error as well as the measurement error
    (if considered) are assumed to be additive.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str], None] = None,
        problem_experiments: Optional[dict] = None,
        additive_measurement_error: bool = False,
        correlation_variables: str = "",
        correlation_model: str = "exp",
        correlation_dict: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        """
        For a detailed explanation of the input arguments check out the docstring given
        in probeye/definition/likelihood_models.py:GaussianLikelihoodModel. The only
        additional argument is 'problem_experiments' which is simply a pointer to
        InferenceProblem._experiments (a dictionary of all the problem's experiments).
        """

        # initialize the super-class (GaussianLikelihoodModel) based on the given input
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            correlation_dict=correlation_dict,
            name=name,
        )

        # problem_experiments is an attribute not set by the parent-class
        self.problem_experiments = problem_experiments  # type: ignore

        # extract the values of the correlation variable
        self.coords = self.coordinate_array(correlation_variables)

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log(likelihood) of this model.
        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.
        prms
            Dictionary containing parameter name:value pairs.
        worst_value
            This value is returned when this method does not result in a numeric value.
            This might happen for example when the given parameters are not valid (for
            example in case of a negative standard deviation). The returned value in
            such cases should represent the worst possible value of the contribution.
        Returns
        -------
        ll
            A scalar value representing the evaluated log-likelihood function.
        """

        # compute the model residuals via a method from the parent class
        res_vector = self.residuals_vector(model_response_dict)
        n = len(res_vector)

        # parameters for the model prediction error
        std_model = prms["std_model"]
        l_corr = prms["l_corr"]
        if (std_model <= 0) or (l_corr <= 0):
            return worst_value

        # parameter for the measurement error
        if self.additive_measurement_error:
            std_meas = prms["std_measurement"]
            if std_meas <= 0:
                return worst_value
        else:
            std_meas = None

        # assemble covariance matrix
        f_corr = lambda a: correlation_function(d=a, correlation_length=l_corr)
        cov_matrix = std_model ** 2 * correlation_matrix(self.coords, f_corr)
        if self.additive_measurement_error:
            cov_matrix += std_meas ** 2 * np.eye(n)

        # evaluate log-likelihood (no efficient algorithm available in this case)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        _, log_det_cov_matrix = np.linalg.slogdet(cov_matrix)
        ll = -(n * np.log(2 * np.pi) + log_det_cov_matrix) / 2
        ll += -np.dot(res_vector, inv_cov_matrix.dot(res_vector)) / 2

        return ll


class AdditiveSpaceTimeCorrelatedModelError1D(GaussianLikelihoodModel):
    """
    This is a likelihood model based on a multivariate normal distribution with that
    includes correlation effects in time and one spatial coordinate. Both the model
    error as well as the measurement error (if considered) are assumed to be additive.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str], None] = None,
        problem_experiments: Optional[dict] = None,
        additive_measurement_error: bool = False,
        correlation_variables: str = "",
        correlation_model: str = "exp",
        correlation_dict: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        """
        For a detailed explanation of the input arguments check out the docstring given
        in probeye/definition/likelihood_models.py:GaussianLikelihoodModel. The only
        additional argument is 'problem_experiments' which is simply a pointer to
        InferenceProblem._experiments (a dictionary of all the problem's experiments).
        """

        # initialize the super-class (GaussianLikelihoodModel) based on the given input
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            correlation_dict=correlation_dict,
            name=name,
        )

        # problem_experiments is an attribute not set by the parent-class
        self.problem_experiments = problem_experiments  # type: ignore

        # extract the values for time
        time_vector = self.coordinate_array("t")
        self.time_vector, self.f_t, self.sorted_time = incrementalize(time_vector)

        # extract the values for the spatial coordinate
        space_coordinate = self.correlation_variables.replace("t", "")
        space_vector = self.coordinate_array(space_coordinate)
        self.space_vector, self.f_x, self.sorted_space = incrementalize(space_vector)

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log(likelihood) of this model.
        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.
        prms
            Dictionary containing parameter name:value pairs.
        worst_value
            This value is returned when this method does not result in a numeric value.
            This might happen for example when the given parameters are not valid (for
            example in case of a negative standard deviation). The returned value in
            such cases should represent the worst possible value of the contribution.
        Returns
        -------
        ll
            A scalar value representing the evaluated log-likelihood function.
        """

        # compute the model residuals via a method from the parent class
        y_model = self.response_array(model_response_dict)
        res_vector = self.residuals_vector(model_response_dict)
        ones = np.ones(len(res_vector))
        zeros = np.zeros(len(res_vector))

        # parameters for the model prediction error
        std_model = prms["std_model"]
        l_corr_x = prms["l_corr_space"]
        l_corr_t = prms["l_corr_time"]
        if (std_model <= 0) or (l_corr_x <= 0) or (l_corr_t <= 0):
            return worst_value
        std_model *= np.ones(len(self.space_vector))

        # parameter for the measurement error
        if self.additive_measurement_error:
            std_meas = prms["std_measurement"]
            if std_meas <= 0:
                return worst_value
        else:
            std_meas = 0

        # efficient log-likelihood evaluation via tripy
        ll = kron_loglike_2D_tridiag(
            y_model,
            self.space_vector,
            self.time_vector,
            std_meas,
            l_corr_x,
            std_model,
            l_corr_t,
            np.ones(len(self.time_vector)),
        )
        return ll


class MultiplicativeUncorrelatedModelError(GaussianLikelihoodModel):
    """
    This is a likelihood model based on a multivariate normal distribution without any
    correlations, i.e., with a diagonal covariance matrix. The model error is assumed to
    be multiplicative while the measurement error (if considered) is assumed additive.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str], None] = None,
        problem_experiments: Optional[dict] = None,
        additive_measurement_error: bool = False,
        correlation_variables: str = "",
        correlation_model: str = "exp",
        correlation_dict: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        """
        For a detailed explanation of the input arguments check out the docstring given
        in probeye/definition/likelihood_models.py:GaussianLikelihoodModel. The only
        additional argument is 'problem_experiments' which is simply a pointer to
        InferenceProblem._experiments (a dictionary of all the problem's experiments).
        """

        # initialize the super-class (GaussianLikelihoodModel) based on the given input
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            correlation_dict=correlation_dict,
            name=name,
        )

        # problem_experiments is an attribute not set by the parent-class
        self.problem_experiments = problem_experiments  # type: ignore

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log(likelihood) of this model.
        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.
        prms
            Dictionary containing parameter name:value pairs.
        worst_value
            This value is returned when this method does not result in a numeric value.
            This might happen for example when the given parameters are not valid (for
            example in case of a negative standard deviation). The returned value in
            such cases should represent the worst possible value of the contribution.
        Returns
        -------
        ll
            A scalar value representing the evaluated log-likelihood function.
        """
        # compute the model error; note that this mode has exactly one sensor
        y_model = self.response_vector(model_response_dict)
        delta = self.residuals_vector(model_response_dict)
        n = len(delta)

        # process the standard deviation for the model error
        std_model = prms["std_model"]
        if std_model <= 0:
            return worst_value

        # compute the covariance matrix depending on the measurement error; note that
        # without correlation (as it is the case here) the covariance matrix is diagonal
        # and can hence be represented by a single vector; this is the case in the
        # computations below, i.e., cov_mtx is a vector
        if self.additive_measurement_error:
            std_meas = prms["std_measurement"]
            if std_meas <= 0:
                return worst_value
            cov_mtx = np.power(y_model * std_model, 2) + np.power(std_meas, 2)
        else:
            cov_mtx = np.power(y_model * std_model, 2)

        # finally, evaluate the log-likelihood
        inv_cov_mtx = 1 / cov_mtx
        log_det_cov_mtx = np.sum(np.log(cov_mtx))
        ll = -0.5 * (n * np.log(2 * np.pi) + log_det_cov_mtx)
        ll += -0.5 * np.sum(np.power(delta, 2) * inv_cov_mtx)
        return ll


class MultiplicativeCorrelatedModelError1D(GaussianLikelihoodModel):
    """
    This is a likelihood model based on a multivariate normal distribution with that
    includes correlation effects in time or in one spatial coordinate. The model error
    is assumed to be multiplicative while the measurement error (if considered) is
    assumed to be additive.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str], None] = None,
        problem_experiments: Optional[dict] = None,
        additive_measurement_error: bool = False,
        correlation_variables: str = "",
        correlation_model: str = "exp",
        correlation_dict: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        """
        For a detailed explanation of the input arguments check out the docstring given
        in probeye/definition/likelihood_models.py:GaussianLikelihoodModel. The only
        additional argument is 'problem_experiments' which is simply a pointer to
        InferenceProblem._experiments (a dictionary of all the problem's experiments).
        """

        # initialize the super-class (GaussianLikelihoodModel) based on the given input
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            correlation_dict=correlation_dict,
            name=name,
        )

        # problem_experiments is an attribute not set by the parent-class
        self.problem_experiments = problem_experiments  # type: ignore

        # extract the values of the correlation variable
        coords = self.coordinate_array(correlation_variables)
        self.coords, self.f, self.sorted_coords = incrementalize(coords)

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log(likelihood) of this model.
        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.
        prms
            Dictionary containing parameter name:value pairs.
        worst_value
            This value is returned when this method does not result in a numeric value.
            This might happen for example when the given parameters are not valid (for
            example in case of a negative standard deviation). The returned value in
            such cases should represent the worst possible value of the contribution.
        Returns
        -------
        ll
            A scalar value representing the evaluated log-likelihood function.
        """

        # compute the model residuals via a method from the parent class
        y_model = self.response_vector(model_response_dict)
        res_vector = self.residuals_vector(model_response_dict)
        if not self.sorted_coords:
            y_model = self.f(y_model)
            res_vector = self.f(res_vector)
        ones = np.ones(len(res_vector))

        # parameters for the model prediction error
        std_model = prms["std_model"]
        l_corr = prms["l_corr"]
        if (std_model <= 0) or (l_corr <= 0):
            return worst_value
        std_model *= ones

        # parameter for the measurement error
        if self.additive_measurement_error:
            std_meas = prms["std_measurement"]
            if std_meas <= 0:
                return worst_value
            std_meas *= ones
        else:
            std_meas = None

        # efficient log-likelihood evaluation via tripy
        ll = chol_loglike_1D(
            self.coords, res_vector, l_corr, std_model, std_meas, y_model=y_model
        )
        return ll


class MultiplicativeSpaceCorrelatedModelError2D3D(GaussianLikelihoodModel):
    """
    This is a likelihood model based on a multivariate normal distribution with that
    includes correlation effects in more than one spatial coordinate. Time correlation
    effects cannot be considered. The model error is assumed to be multiplicative while
    the measurement error (if considered) is assumed to be additive.
    """

    def __init__(
        self,
        prms_def: Union[str, List[Union[str, dict]], dict],
        sensors: Union["Sensor", List["Sensor"]],
        experiment_names: Union[str, List[str], None] = None,
        problem_experiments: Optional[dict] = None,
        additive_measurement_error: bool = False,
        correlation_variables: str = "",
        correlation_model: str = "exp",
        correlation_dict: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        """
        For a detailed explanation of the input arguments check out the docstring given
        in probeye/definition/likelihood_models.py:GaussianLikelihoodModel. The only
        additional argument is 'problem_experiments' which is simply a pointer to
        InferenceProblem._experiments (a dictionary of all the problem's experiments).
        """

        # initialize the super-class (GaussianLikelihoodModel) based on the given input
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            correlation_dict=correlation_dict,
            name=name,
        )

        # problem_experiments is an attribute not set by the parent-class
        self.problem_experiments = problem_experiments  # type: ignore

        # extract the values of the correlation variable
        self.coords = self.coordinate_array(correlation_variables)

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log(likelihood) of this model.
        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.
        prms
            Dictionary containing parameter name:value pairs.
        worst_value
            This value is returned when this method does not result in a numeric value.
            This might happen for example when the given parameters are not valid (for
            example in case of a negative standard deviation). The returned value in
            such cases should represent the worst possible value of the contribution.
        Returns
        -------
        ll
            A scalar value representing the evaluated log-likelihood function.
        """

        # compute the model residuals via a method from the parent class
        y_model = self.response_vector(model_response_dict)
        res_vector = self.residuals_vector(model_response_dict)
        n = len(res_vector)

        # parameters for the model prediction error
        std_model = prms["std_model"]
        l_corr = prms["l_corr"]
        if (std_model <= 0) or (l_corr <= 0):
            return worst_value

        # parameter for the measurement error
        if self.additive_measurement_error:
            std_meas = prms["std_measurement"]
            if std_meas <= 0:
                return worst_value
        else:
            std_meas = None

        # assemble covariance matrix
        f_corr = lambda a: correlation_function(d=a, correlation_length=l_corr)
        cov_matrix = std_model ** 2 * correlation_matrix(self.coords, f_corr)
        cov_matrix = np.dot(y_model, cov_matrix)
        cov_matrix = np.dot(cov_matrix, y_model)
        if self.additive_measurement_error:
            cov_matrix += std_meas ** 2 * np.eye(n)

        # evaluate log-likelihood (no efficient algorithm available in this case)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        _, log_det_cov_matrix = np.linalg.slogdet(cov_matrix)
        ll = -(n * np.log(2 * np.pi) + log_det_cov_matrix) / 2
        ll += -np.dot(res_vector, inv_cov_matrix.dot(res_vector)) / 2

        return ll


def translate_likelihood_model(
    like_def: GaussianLikelihoodModel,
) -> Union[
    AdditiveUncorrelatedModelError,
    AdditiveCorrelatedModelError1D,
    AdditiveSpaceCorrelatedModelError2D3D,
    MultiplicativeUncorrelatedModelError,
    MultiplicativeCorrelatedModelError1D,
    MultiplicativeSpaceCorrelatedModelError2D3D,
]:
    """
    Translates a given instance of GaussianLikelihoodModel (which is essentially just a
    description of the likelihood model without any computing-methods) to a specific
    likelihood model object which does contain SciPy-based computing-methods.
    Parameters
    ----------
    like_def
        An instance of GaussianLikelihoodModel which contains general information on the
        likelihood model but no computing-methods.
    Returns
    -------
    likelihood_computer
        An instance of a specific likelihood model class with computing capabilities.
        Examples for such classes are given above in this file.
    """

    # likelihood model selection based on the flags given in the likelihood definition
    if like_def.additive_model_error:
        if not like_def.considers_correlation:
            l_class = AdditiveUncorrelatedModelError  # type: ignore
        else:
            if like_def.considers_only_space_correlation:
                if len(like_def.correlation_variables) == 1:
                    l_class = AdditiveCorrelatedModelError1D  # type: ignore
                else:
                    l_class = AdditiveSpaceCorrelatedModelError2D3D  # type: ignore
            elif like_def.considers_only_time_correlation:
                l_class = AdditiveCorrelatedModelError1D  # type: ignore
            else:
                if len(like_def.correlation_variables) == 2:
                    l_class = AdditiveSpaceTimeCorrelatedModelError1D  # type: ignore
                else:
                    raise NotImplementedError("Likelihood model not implemented yet!")
    else:
        if not like_def.considers_correlation:
            l_class = MultiplicativeUncorrelatedModelError  # type: ignore
        else:
            if like_def.considers_only_space_correlation:
                if len(like_def.correlation_variables) == 1:
                    l_class = MultiplicativeCorrelatedModelError1D  # type: ignore
                else:
                    l_class = MultiplicativeSpaceCorrelatedModelError2D3D  # type: ignore
            elif like_def.considers_only_time_correlation:
                l_class = MultiplicativeCorrelatedModelError1D  # type: ignore
            else:
                raise NotImplementedError("Likelihood model not implemented yet!")

    # this is where the translation happens
    likelihood_computer = l_class(
        like_def.prms_def,
        like_def.sensors,
        experiment_names=like_def.experiment_names,
        problem_experiments=like_def.problem_experiments,
        additive_measurement_error=like_def.additive_measurement_error,
        correlation_variables=like_def.correlation_variables,
        correlation_model=like_def.correlation_model,
        correlation_dict=like_def.correlation_dict,
        name=like_def.name,
    )

    return likelihood_computer
