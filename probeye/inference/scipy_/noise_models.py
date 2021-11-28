# standard library imports
from typing import Union, List, Optional, TYPE_CHECKING

# third party imports
import numpy as np

# local imports
from probeye.subroutines import len_or_one
from probeye.definition.noise_model import NormalNoiseModel
from probeye.inference.scipy_.correlation_models import (
    SpatiotemporalExponentialCorrelationModel,
)

# imports only needed for type hints
if TYPE_CHECKING:  # pragma: no cover
    from probeye.definition.sensor import Sensor

class NormalNoise(NormalNoiseModel):
    def __init__(
        self,
        prms_def,
        sensors,
        experiment_names=None,
        problem_experiments=None,
        name=None,
        corr_static="",
        corr_dynamic="",
        corr_model="exp",
        corr_dict=None,
        noise_type="additive",
    ):
        """
        For a detailed explanation of the input arguments check out the docstring given
        in probeye/definition/noise_models.py:NoiseModelBase. The only additional
        argument is 'problem_experiments' which is a pointer to InferenceProblem.
        _experiments (a dictionary of all the problem's experiments).
        """

        # initialize the super-class (NormalNoiseModel) based on the given input
        super().__init__(
            prms_def,
            sensors,
            experiment_names=experiment_names,
            name=name,
            corr_static=corr_static,
            corr_dynamic=corr_dynamic,
            corr_model=corr_model,
            corr_dict=corr_dict,
            noise_type=noise_type,
        )

        # problem_experiments is an attribute not set by the parent-class
        self.problem_experiments = problem_experiments

        # the two attributes n_static and n_dynamic contain the number of
        # rows/columns the distance array would need to have if there were
        # only the provided static/dynamic correlation variables; both numbers
        # must be either identical, or - if they are not - one of them must be
        # zero (e.g. (8, 8), (0, 9), (17, 0) but not e.g. (18, 11))
        self.n_static, self.n_dynamic = self.process_correlation_definition()
        self.n = max(self.n_static, self.n_dynamic)

        if self.corr:
            # in this case, correlation is considered in the noise model
            self.position_arrays = self.generate_distance_array()
            if self.corr_model == "exp":
                self.cov = SpatiotemporalExponentialCorrelationModel(
                    self.position_arrays
                )
                self.loglike_contribution = self.loglike_contribution_with_correlation
            else:
                raise ValueError(
                    f"Encountered unknown flag '{self.corr_model}' "
                    f"for requested correlation model.\n Currently available "
                    f"options are: 'exp' for an exponential model."
                )
        else:
            # in this case, no correlation is considered in the noise model
            self.cov = None
            self.loglike_contribution = self.loglike_contribution_without_correlation

    def process_correlation_definition(self):

        # the following variables will be derived below
        n_static = 0
        n_dynamic = 0

        if self.corr_static:
            # check that each of the noise model's sensors has the specified
            # correlation variables as attributes
            first_exp_name = self.experiment_names[0]
            for v in self.corr_static:
                attribute = self.corr_dict[first_exp_name][v]
                sensor_value = None
                for sensor in self.sensors:
                    if not hasattr(sensor, attribute):
                        raise AttributeError(
                            f"Sensor '{sensor.name}' does not have an "
                            f"attribute '{attribute}' as implied by the "
                            f"correlation definition."
                        )
                    sensor_value_new = getattr(sensor, attribute)
                    if sensor_value is not None:
                        if sensor_value != sensor_value_new:
                            raise RuntimeError(
                                f"The static correlation variable '{v}' "
                                f"varies between the sensors of the noise "
                                f"model. They must be similar."
                            )

            # compute how many columns (rows) the quadratic distance array must
            # have if there was only the static and no dynamic correlation data
            attribute = self.corr_dict[first_exp_name][self.corr_static[0]]
            corr_data_length = len_or_one(getattr(self.sensors[0], attribute))
            n_static = corr_data_length * self.n_experiments * self.n_sensors

        if self.corr_dynamic:
            # check that each experiment assigned to this noise model has
            # the specified dynamic correlation variables as sensor values
            for exp_name in self.experiment_names:
                experiment = self.problem_experiments[exp_name]
                sensor_values = experiment["sensor_values"]
                for v in self.corr_dynamic:
                    key = self.corr_dict[exp_name][v]
                    if key not in sensor_values:
                        raise KeyError(
                            f"Experiment '{exp_name}' does not contain sensor "
                            f"values for '{key}' (alias of '{v}') as implied "
                            f"by the correlation definition."
                        )
            # compute how many columns (rows) the quadratic distance array must
            # have if there was only the dynamic and no static correlation data
            for exp_name in self.experiment_names:
                experiment = self.problem_experiments[exp_name]
                sensor_values = experiment["sensor_values"]
                v = sensor_values[self.corr_dict[exp_name][self.corr_dynamic[0]]]
                n_dynamic_exp = len_or_one(v)
                if n_static > 0:
                    if n_dynamic_exp != n_static:
                        raise RuntimeError(
                            f"Encountered dynamic correlation data "
                            f"'{self.corr_dynamic[0]}' in experiment '"
                            f"{exp_name}' with a length {n_dynamic_exp} which "
                            f"does not agree with the length of the static "
                            f"correlation variables {n_static} defined within "
                            f"the noise model."
                        )
                n_dynamic += len_or_one(v) * self.n_sensors

        return n_static, n_dynamic

    def generate_distance_array(self):
        """

        """

        position_arrays = {v: np.zeros((self.n, self.n)) for v in list(self.corr)}
        for v in self.corr:
            idx = 0
            for exp_name in self.experiment_names:
                if v in self.corr_dynamic:
                    experiment = self.problem_experiments[exp_name]
                    key = self.corr_dict[exp_name][v]
                    data = experiment["sensor_values"][key]
                else:
                    attribute = self.corr_dict[exp_name][v]
                    data = getattr(self.sensors[0], attribute)
                m = len_or_one(data)
                for sensor in self.sensors:
                    position_arrays[v][idx : idx + m, :] = np.tile(
                        data.reshape((m, -1)), self.n
                    )
                    idx += m
        return position_arrays

    def loglike_contribution_without_correlation(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        This method overwrites the corresponding method of the parent class. Check out
        the docstring there for additional information.
        """
        # compute the model error; note that this mode has exactly one sensor
        model_error_vector = self.error_vector(model_response)
        # the precision 'prec' is defined as the inverse of the variance, hence
        # prec = 1 / sigma**2 where sigma denotes the standard deviation
        std = prms["std"]
        mean = 0.0 if self.zero_mean else prms["mean"]
        prec = 1.0 / std ** 2.0
        # evaluate the Gaussian log-PDF with zero mean and a variance of 1/prec for
        # each error term and sum them up
        ll = -len(model_error_vector) / 2 * np.log(2 * np.pi / prec)
        ll -= 0.5 * prec * np.sum(np.square(model_error_vector - mean))
        return ll

    def loglike_contribution_with_correlation(
        self, model_response_dict: dict, prms: dict, worst_value: float = -np.infty
    ) -> float:
        """
        Evaluates the log-likelihood for the noise model's experiments.

        Parameters
        ----------
        model_response_dict : dict
            The keys are the names of the experiments. The values are dicts
            which contain the forward model's output sensor's names as keys
            and have the corresponding model responses as values.
        prms : dict
            Contains the names of the correlation model's parameters as keys
            and the corresponding numeric values as values.
        worst_value : float, int, optional
            This value is returned when this method does not result in a numeric
            value. This might happen for example when the given parameters are
            not valid (for example in case of a negative standard deviation).
            The returned value in such cases should represent the worst possible
            value of the contribution.

        Returns
        -------
        ll
            The evaluated log-likelihood function.
        """
        # before computing the covariance matrix, check if the given parameter values
        # are valid; if not, the computation of the log-like contribution is stopped
        # right away while returning the worst value its evaluation can result in
        if not self.cov.check_prms(prms):
            return worst_value

        cov_matrix = self.cov(prms)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        n = cov_matrix.shape[0]
        _, log_det_cov_matrix = np.linalg.slogdet(cov_matrix)
        error_vector = self.error_vector(model_response_dict)
        error_vector_red = np.dot(self.cov.reduction_array, error_vector)
        ll = -0.5 * (n * np.log(2 * np.pi) + log_det_cov_matrix +
                     np.dot(error_vector_red, inv_cov_matrix.dot(error_vector_red)))
        return ll


def translate_noise_model(noise_base):
    """
    Translates a given instance of NoiseBase (which is essentially just a description
    of the noise model without computing-methods) to a specific noise model object which
    does contain computing-methods (e.g. compute the log-likelihood contribution).

    Parameters
    ----------
    noise_base : obj[NoiseBase]
        An instance of NoiseBase (or the child-class NormalNoiseModel) which contains
        basic information on the noise model but no computing-methods.

    Returns
    -------
    noise_object
        An instance of a specific noise model class with computing capabilities.
        Examples for such classes are given above in this file.
    """

    # this is the noise classes currently defined (see code above)
    noise_classes = {"normal": NormalNoise}

    # this is where the translation happens
    noise_object = noise_classes[noise_base.dist](
        noise_base.prms_def,
        noise_base.sensors,
        experiment_names=noise_base.experiment_names,
        problem_experiments=noise_base.problem_experiments,
        name=noise_base.name,
        corr_static=noise_base.corr_static,
        corr_dynamic=noise_base.corr_dynamic,
        corr_model=noise_base.corr_model,
        corr_dict=noise_base.corr_dict,
        noise_type=noise_base.noise_type,
    )

    return noise_object
