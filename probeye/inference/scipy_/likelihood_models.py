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
from probeye.subroutines import len_or_one

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
        in probeye/definition/likelihood_models.py:GaussianLikelihoodModel.
        """
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            correlation_dict=correlation_dict,
            name=name,
        )

    def residuals(self, model_response_dict: dict) -> dict:
        """
        Computes the model residuals (model prediction minus measurement) for all of the
        likelihood model's experiments and returns them as a dictionary.

        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.

        Returns
        -------
        residuals_dict
            A dictionary with the keys being the likelihood model's sensor names, and 1D
            numpy arrays representing the model residuals for all experiments as values.
        """
        # prepare the dictionary keys
        residuals_dict = {name: np.array([]) for name in self.sensor_names}

        # fill the dictionary with model residual vectors
        for exp_name in self.experiment_names:
            exp_dict = self.problem_experiments[exp_name]  # type: ignore
            ym_dict = model_response_dict[exp_name]
            ye_dict = exp_dict["sensor_values"]
            residuals_dict = {
                name: np.append(residuals_dict[name], ym_dict[name] - ye_dict[name])
                for name in self.sensor_names
            }

        return residuals_dict

    def residuals_vector(self, model_response_dict: dict) -> np.ndarray:
        """
        Computes the model residuals for all of the likelihood model's sensors over all
        of the likelihood model's experiments and returns them in a single vector.

        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.

        Returns
        -------
        residuals_vector
            A one-dimensional vector containing the model residuals.
        """
        residuals_dict = self.residuals(model_response_dict)
        n = 0
        for residuals_sub_vector in residuals_dict.values():
            n += len_or_one(residuals_sub_vector)
        residuals_vector = np.zeros(n)
        idx = 0
        for residuals_sub_vector in residuals_dict.values():
            m = len_or_one(residuals_sub_vector)
            residuals_vector[idx : idx + m] = residuals_sub_vector
            idx += m
        return residuals_vector

    def response_vector(self, model_response_dict: dict) -> np.ndarray:
        """
        Computes the model response for all of the likelihood model's sensors over all
        of the likelihood model's experiments and returns them in a single vector.

        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.

        Returns
        -------
        response_vector
            A one-dimensional vector containing the model response.
        """

        # create the response dict only for the experiments of the likelihood model
        response_dict = {name: np.array([]) for name in self.sensor_names}
        for exp_name in self.experiment_names:
            ym_dict = model_response_dict[exp_name]
            response_dict = {
                name: np.append(response_dict[name], ym_dict[name])
                for name in self.sensor_names
            }

        # concatenate the responses to a single vector
        n = 0
        for residuals_sub_vector in response_dict.values():
            n += len_or_one(residuals_sub_vector)
        residuals_vector = np.zeros(n)
        idx = 0
        for residuals_sub_vector in response_dict.values():
            m = len_or_one(residuals_sub_vector)
            residuals_vector[idx : idx + m] = residuals_sub_vector
            idx += m
        return residuals_vector

    def response_array(self, model_response_dict: dict) -> np.ndarray:
        """
        Computes the model response for all of the likelihood model's sensors over all
        of the likelihood model's experiments and returns them in an array format.

        Parameters
        ----------
        model_response_dict
            The first key is the name of the experiment. The values are dicts which
            contain the forward model's output sensor's names as keys have the
            corresponding model responses as values.

        Returns
        -------
        res_array
            An 2D array containing the model response.
        """

        # prepare the result array
        n_rows = self.n_experiments * self.n_sensors
        n_cols = len_or_one(
            model_response_dict[self.experiment_names[0]][self.sensor_names[0]]
        )
        res_array = np.zeros((n_rows, n_cols))

        # create the response dict only for the experiments of the likelihood model
        i = 0
        for experiment_name in self.experiment_names:
            for sensor_name in self.sensor_names:
                res_array[i, :] = model_response_dict[experiment_name][sensor_name]
                i += 1

        return res_array

    def coordinate_array(self, coords: str) -> np.ndarray:
        """
        Parameters
        ----------
        coords
            One or more characters from {'x', 'y', 'z', 't'}, i.e., possible correlation
            variables.

        Returns
        -------
        coord_array
            A vector containing the values of the requested coordinates 'coords' over
            all of the likelihood model's experiments and sensors. These values have
            the same structure as the values in the residual vector computed by the
            method 'residuals_vector' above. Hence, the i-th entry in 'coord_array'
            corresponds to the i-th entry in the vector returned by 'residuals_vector'.
        """

        # check input
        n_coords = len(coords)
        coord_1 = list(coords)[0]

        # prepare the coord-vector with the correct length
        n = 0
        ns = len(self.sensors)
        for exp_name in self.experiment_names:
            exp_dict = self.problem_experiments[exp_name]  # type: ignore
            exp_sensor_values = exp_dict["sensor_values"]
            coord_name_in_exp = self.correlation_dict[exp_name][coord_1]  # type: ignore
            if coord_name_in_exp in exp_sensor_values:
                n += len_or_one(exp_sensor_values[coord_name_in_exp]) * ns
            else:
                for sensor in self.sensors:
                    try:
                        n += len_or_one(getattr(sensor, coord_1))
                    except AttributeError:
                        print(
                            f"Sensor '{sensor.name}' of likelihood model "
                            f"'{self.name}' does not have a '{coord_1}'-attribute!"
                        )
                        raise
        coord_array = np.zeros((n, n_coords))

        # fill the dictionary with model residual vectors
        for ic, coord in enumerate(list(coords)):
            i = 0
            for exp_name in self.experiment_names:
                exp_dict = self.problem_experiments[exp_name]  # type: ignore
                exp_sensor_values = exp_dict["sensor_values"]
                coord_in_exp = self.correlation_dict[exp_name][coord]  # type: ignore
                if coord_in_exp in exp_sensor_values:
                    coord_sub_vector = exp_sensor_values[coord_in_exp]
                    m = len_or_one(coord_sub_vector)
                    for _ in range(ns):
                        coord_array[i : i + m, ic] = coord_sub_vector
                        i += m
                else:
                    for sensor in self.sensors:
                        coord_sub_vector = getattr(sensor, coord)
                        m = len_or_one(coord_sub_vector)
                        coord_array[i : i + m, ic] = coord_sub_vector
                        i += m

        if n_coords == 1:
            coord_array = coord_array.flatten()

        return coord_array

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes and returns the log(likelihood). To be overwritten by one of the
        specific classes listed below.

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
        raise NotImplementedError(
            "The ScipyLikelihoodBase class does not have a loglike implementation."
        )


class AdditiveUncorrelatedModelError(ScipyLikelihoodBase):
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
        in probeye/definition/likelihood_models.py:GaussianLikelihoodModel.
        """

        # initialize the super-class (ScipyLikelihoodBase) based on the given input
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            correlation_dict=correlation_dict,
            name=name,
        )

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log(likelihood) of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
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


class AdditiveCorrelatedModelError1D(ScipyLikelihoodBase):
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
        in probeye/definition/likelihood_models.py:GaussianLikelihoodModel.
        """

        # initialize the super-class (ScipyLikelihoodBase) based on the given input
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            correlation_dict=correlation_dict,
            name=name,
        )

        # extract the values of the correlation variable
        coords = self.coordinate_array(correlation_variables)
        self.coords, self.f, self.sorted_coords = incrementalize(coords)

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log(likelihood) of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
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


class AdditiveSpaceCorrelatedModelError2D3D(ScipyLikelihoodBase):
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
        in probeye/definition/likelihood_models.py:GaussianLikelihoodModel.
        """

        # initialize the super-class (ScipyLikelihoodBase) based on the given input
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            correlation_dict=correlation_dict,
            name=name,
        )

        # extract the values of the correlation variable
        self.coords = self.coordinate_array(correlation_variables)

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log(likelihood) of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
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


class AdditiveSpaceTimeCorrelatedModelError1D(ScipyLikelihoodBase):
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
        in probeye/definition/likelihood_models.py:GaussianLikelihoodModel.
        """

        # initialize the super-class (ScipyLikelihoodBase) based on the given input
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            correlation_dict=correlation_dict,
            name=name,
        )

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
        Computes the log(likelihood) of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
        """

        # compute the model residuals via a method from the parent class
        y_model = self.response_array(model_response_dict)
        # res_vector = self.residuals_vector(model_response_dict)

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


class MultiplicativeUncorrelatedModelError(ScipyLikelihoodBase):
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
        in probeye/definition/likelihood_models.py:GaussianLikelihoodModel.
        """

        # initialize the super-class (ScipyLikelihoodBase) based on the given input
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            correlation_dict=correlation_dict,
            name=name,
        )

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log(likelihood) of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
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


class MultiplicativeCorrelatedModelError1D(ScipyLikelihoodBase):
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

        # initialize the super-class (ScipyLikelihoodBase) based on the given input
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            correlation_dict=correlation_dict,
            name=name,
        )

        # extract the values of the correlation variable
        coords = self.coordinate_array(correlation_variables)
        self.coords, self.f, self.sorted_coords = incrementalize(coords)

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log(likelihood) of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
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


class MultiplicativeSpaceCorrelatedModelError2D3D(ScipyLikelihoodBase):
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

        # initialize the super-class (ScipyLikelihoodBase) based on the given input
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            problem_experiments=problem_experiments,
            additive_measurement_error=additive_measurement_error,
            correlation_variables=correlation_variables,
            correlation_model=correlation_model,
            correlation_dict=correlation_dict,
            name=name,
        )

        # extract the values of the correlation variable
        self.coords = self.coordinate_array(correlation_variables)

    def loglike(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Computes the log(likelihood) of this model. For more information, check out the
        doc-string of the parent class (ScipyLikelihoodBase).
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
    if lm_def.additive_model_error:
        if not lm_def.considers_correlation:
            l_class = "Add_Uncorrelated"
        else:
            if lm_def.considers_only_space_correlation:
                if len(lm_def.correlation_variables) == 1:
                    l_class = "Add_Correlated_1D"
                else:
                    l_class = "Add_SpaceCorrelated_2D3D"
            elif lm_def.considers_only_time_correlation:
                l_class = "Add_Correlated_1D"
            else:
                if len(lm_def.correlation_variables) == 2:
                    l_class = "Add_SpaceTimeCorrelated_1D"
                else:
                    l_class = "Add_SpaceTimeCorrelated_2D3D"
    else:
        if not lm_def.considers_correlation:
            l_class = "Mul_Uncorrelated"
        else:
            if lm_def.considers_only_space_correlation:
                if len(lm_def.correlation_variables) == 1:
                    l_class = "Mul_Correlated_1D"
                else:
                    l_class = "Mul_SpaceTimeCorrelated_2D3D"
            elif lm_def.considers_only_time_correlation:
                l_class = "Mul_Correlated_1D"
            else:
                if len(lm_def.correlation_variables) == 2:
                    l_class = "Mul_SpaceTimeCorrelated_1D"
                else:
                    l_class = "Mul_SpaceTimeCorrelated_2D3D"

    # this dict allows to map an assigned string from the if-cases above to a specific
    # likelihood model defined in this file; the class is not assigned directly to the
    # variable l_class from the if-clauses above to avoid that l_class can have multiple
    # different types (which leads to issues during type-checking)
    class_dict = {
        "Add_Uncorrelated": AdditiveUncorrelatedModelError,
        "Add_Correlated_1D": AdditiveCorrelatedModelError1D,
        "Add_SpaceCorrelated_2D3D": AdditiveSpaceCorrelatedModelError2D3D,
        "Add_SpaceTimeCorrelated_1D": AdditiveSpaceTimeCorrelatedModelError1D,
        # "Add_SpaceTimeCorrelated_2D3D": AdditiveSpaceTimeCorrelatedModelError2D3D,
        "Mul_Uncorrelated": MultiplicativeUncorrelatedModelError,
        "Mul_Correlated_1D": MultiplicativeCorrelatedModelError1D,
        "Mul_SpaceCorrelated_2D3D": MultiplicativeSpaceCorrelatedModelError2D3D,
        # "Mul_SpaceTimeCorrelated_1D": MultiplicativeSpaceTimeCorrelatedModelError1D,
        # "Mul_SpaceTimeCorrelated_2D3D": MultiplicativeSpaceTimeCorrelatedModelError2D3D,
    }

    # this is where the translation happens
    likelihood_computer = class_dict[l_class](
        lm_def.prms_def,
        lm_def.sensors,
        experiment_names=lm_def.experiment_names,
        problem_experiments=lm_def.problem_experiments,
        additive_measurement_error=lm_def.additive_measurement_error,
        correlation_variables=lm_def.correlation_variables,
        correlation_model=lm_def.correlation_model,
        correlation_dict=lm_def.correlation_dict,
        name=lm_def.name,
    )

    return likelihood_computer
