# third party imports
import numpy as np

# local imports
from probeye.definition.noise_model import NormalNoiseModel
from probeye.inference.scipy_.correlation_models import (
    SpatialExponentialCorrelationModel,
)


class NormalNoise(NormalNoiseModel):
    def __init__(
        self,
        target_sensor,
        prms_def,
        name=None,
        corr=None,
        corr_model="exp",
        noise_type="additive",
    ):
        """
        Parameters
        ----------
        target_sensor : obj[Sensor]
            The sensor, the noise refers to. Note that this is not a sensor's
            name but a Sensor object. This object must contain the spatial
            information that is referred to in corr. For example, if corr is set
            to 'xy' then target_sensor must have a x and a y attribute that
            contain coordinate vectors of similar lengths.
        prms_def : {str, list, dict}
            Contains the noise model's parameter names. The list may only
            contain strings or one-element dictionaries. It could look, for
            example, like [{'a': 'm'}, 'b']. The one-element dictionaries
            account for the possibility to define a local name for a latent
            parameter that is different from the global name. In the example
            above, the latent parameter with the global name 'a' will be
            referred to as 'm' within the noise model. So, the one-element dicts
            have the meaning {<global name>: <local name>}. String-elements are
            interpreted as having similar local and global names. Note that the
            local-name option will not be required most of the times. The input
            from global to local name can also be provided as a dict. In the
            example above it would look like {'a': 'm', 'b': 'b'}.
        name : {str, None}, optional
            Unique name of the noise model. This name is None, if the user does
            not specify it when adding the noise model to the problem. It is
            then named automatically before starting the inference engine.
        corr : {'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz', None}, optional
            Spatial coordinates to be considered when setting up the covariance
            matrix. For example, when 'x' is chosen, the distance between two
            points i and j is computed as |xi - xj|. If 'xz' is chosen, this
            distance would be computed as sqrt((xi-xj)**2 + (zi-zj)**2). If None
            is chosen, no spatial correlation will be considered.
        corr_model : {'exp'}, optional
            Defines the correlation function the covariance matrix will be based
            on. Currently, there is only the one optional of an exponential
            model ('exp'). More might be added in future versions. This argument
            has no effect if corr is set to None.
        noise_type : {'additive', 'multiplicative', 'other'}, optional
            Defines if the model error is computed by [prediction - measurement]
            ('additive') or via [prediction/measurement-1] ('multiplicative') or
            in some 'other' i.e., non-standard fashion.
        """

        # initialize the super-class (NormalNoiseModel) based on the given input
        super().__init__(
            prms_def=prms_def,
            sensors=target_sensor,
            name=name,
            corr=corr,
            corr_model=corr_model,
            noise_type=noise_type,
        )

        # the target_sensor is not an attribute of the super-class
        self.target_sensor = target_sensor

        # correlation related attributes
        self.corr = corr
        self.corr_model = corr_model
        if self.corr is not None:
            if self.corr_model == "exp":
                self.cov = SpatialExponentialCorrelationModel(
                    coords=target_sensor.coords, order=list(self.corr)
                )
                self.loglike_contribution = self.loglike_contribution_with_correlation
            else:
                raise ValueError(
                    f"Encountered unknown flag '{self.corr_model}' "
                    f"for requested correlation model.\n Currently available "
                    f"options are: 'exp' for an exponential model."
                )
        else:
            self.loglike_contribution = self.loglike_contribution_without_correlation

    def loglike_contribution_without_correlation(self, model_response, prms):
        """
        This method overwrites the corresponding method of the parent class.
        Check out the docstring there for additional information.
        """
        # compute the model error; note that this mode has exactly one sensor
        model_error_vector = self.error(model_response)[self.target_sensor.name]
        # the precision 'prec' is defined as the inverse of the variance, hence
        # prec = 1 / sigma**2 where sigma denotes the standard deviation
        std = prms["std"]
        mean = 0.0 if self.zero_mean else prms["mean"]
        prec = 1.0 / std ** 2.0
        # evaluate the Gaussian log-PDF with zero mean and a variance of
        # 1/prec for each error term and sum them up
        ll = -len(model_error_vector) / 2 * np.log(2 * np.pi / prec)
        ll -= 0.5 * prec * np.sum(np.square(model_error_vector - mean))
        return ll

    def loglike_contribution_with_correlation(
        self, model_response_dict, prms, worst_value=-np.infty
    ):
        """
        Evaluates the log-likelihood for the noise model's experiments.

        Parameters
        ----------
        model_response_dict : dict
            The first key is the name of the experiment. The values are dicts
            which contain the forward model's output sensor's names as keys
            have the corresponding model responses as values.
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
        ll : float
            The evaluated log-likelihood function.
        """
        # before computing the covariance matrix, check if the given parameter
        # values are valid; if not, the computation of the log-likelihood
        # contribution is stopped right away while returning the worst value
        # its evaluation can result in
        if not self.cov.check_prms(prms):
            return worst_value
        # since the covariance matrix is the same for all experiments in the
        # noise model, values derived from it (its inverse and determinant) are
        # computed before entering the loop over the noise model's experiments
        cov_matrix = self.cov(prms)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        # the following command computes the natural log and the sign of the
        # matrix cov_matrix; note that cov_matrix is positive definite, hence
        # the sign will always be positive and therefore isn't requested here
        _, log_det_cov_matrix = np.linalg.slogdet(cov_matrix)
        n = self.cov.n  # number of data points recorded by the sensor
        n_exp = len(self.experiment_names)
        # the term in the first parenthesis is added by each experiment in the
        # loop; this is why we can pre-compute it outside the loop
        ll = n_exp * (-(n * np.log(2 * np.pi) + log_det_cov_matrix) / 2)
        for exp_name in self.experiment_names:
            exp_dict = self.problem_experiments[exp_name]
            ym = model_response_dict[exp_name]
            ye = exp_dict["sensor_values"]
            error = self.error_function(ym, ye)[self.target_sensor.name]
            ll += -np.dot(error, inv_cov_matrix.dot(error)) / 2
        return ll


def translate_noise_model(noise_base):
    """
    Translates a given instance of NoiseBase (which is essentially just a
    description of the noise model without computing-methods) to a specific
    noise model object which does contain computing-methods (e.g. compute the
    log-likelihood contribution).

    Parameters
    ----------
    noise_base : obj[NoiseBase]
        An instance of NoiseBase which contains basic information on the noise
        model but no computing-methods.

    Returns
    -------
    noise_object : obj[NoiseBase]
        An instance of a specific noise model class with computing capabilities.
        Examples for such classes are given above in this file.
    """

    # this is the noise classes currently defined (see code above)
    noise_classes = {"normal": NormalNoise}

    # this is where the translation happens
    noise_object = noise_classes[noise_base.dist](
        target_sensor=noise_base.sensors[0],
        prms_def=noise_base.prms_def,
        name=noise_base.name,
        corr=noise_base.corr,
        corr_model=noise_base.corr_model,
        noise_type=noise_base.noise_type,
    )

    # here, we take the assigned experiments from the base object
    noise_object.experiment_names = noise_base.experiment_names
    noise_object.problem_experiments = noise_base.problem_experiments

    return noise_object
